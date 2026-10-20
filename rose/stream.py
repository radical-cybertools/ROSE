import asyncio
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

_WAKE = object()  # sentinel: unblocks a waiting window collector
_TIMEOUT = object()  # sentinel: max_wait expired


class Stream:
    """Stream of data items that drives a learner loop.

    Items arrive via :meth:`feed` or attached async-iterator sources and are
    collected into windows of ``batch_size`` items (a partial window is
    flushed after ``max_wait`` seconds). Passing a stream to a learner's
    ``start(stream=...)`` makes each window trigger one learning iteration.

    Every learner consuming the stream receives every item (broadcast).
    Windows are formed per consumer, so they are identical across consumers
    only for ``max_wait=None`` and ``conflate=False``. Items fed before the
    first window is pulled are delivered to all consumers.

    The stream ends once all attached sources are exhausted, or when
    :meth:`close` is called; a stream fed only via :meth:`feed` runs until
    closed or until the consuming learner is stopped. Streams are one-shot.

    Example::

        stream = Stream(batch_size=10, max_wait=2.0)
        stream.attach_source(sensor_stream())

        async for state in learner.start(stream=stream):
            ...
    """

    def __init__(
        self,
        batch_size: int = 1,
        max_wait: float | None = None,
        conflate: bool = False,
        sources: AsyncIterator[Any] | list[AsyncIterator[Any]] | None = None,
    ) -> None:
        """Initialize the stream.

        Args:
            batch_size: Number of items per learning window.
            max_wait: Flush a partial window after this many seconds of
                waiting for more items. None waits for a full window.
            conflate: If True, drop the oldest backlog at ingestion time and
                keep only the newest ~``batch_size`` items when iterations
                are slower than the stream ("latest wins"); queues stay
                bounded.
            sources: One or more async iterators to consume as data
                sources; equivalent to calling :meth:`attach_source` for
                each.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_wait is not None and max_wait <= 0:
            raise ValueError(f"max_wait must be positive or None, got {max_wait}")
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.conflate = conflate

        self._backlog: deque[Any] = deque(maxlen=batch_size if conflate else None)
        self._subs: list[_Subscription] = []
        self._pending_sources: list[AsyncIterator[Any]] = []
        self._pumps: list[asyncio.Task] = []
        self._open_sources = 0
        self._started = False
        self._closed = False
        self._error: BaseException | None = None

        if sources is not None:
            for source in sources if isinstance(sources, list) else [sources]:
                self.attach_source(source)

    async def feed(self, item: Any) -> None:
        """Feed a single data item into the stream."""
        if self._closed:
            raise RuntimeError("feed() on a closed stream")
        self._put(item)

    def attach_source(self, source: AsyncIterator[Any]) -> None:
        """Attach an async iterator as a data source.

        Sources are consumed once the first window is pulled.
        """
        if self._closed:
            raise RuntimeError("attach_source() on a closed stream")
        self._open_sources += 1
        if self._started:
            self._start_pump(source)
        else:
            self._pending_sources.append(source)

    def close(self) -> None:
        """End the stream: cancel sources; consumers drain queued items, then stop."""
        self._closed = True
        self._end()

    async def aclose(self) -> None:
        """Close the stream and wait for the source pumps to finish."""
        self.close()
        await asyncio.gather(*self._pumps, return_exceptions=True)

    def _put(self, item: Any) -> None:
        if not self._started:
            self._backlog.append(item)  # deque maxlen conflates the backlog
            return
        for sub in self._subs:
            sub._put(item)

    def _subscribe(self) -> "_Subscription":
        if self._started:
            raise RuntimeError("cannot subscribe after the first window was pulled")
        sub = _Subscription(self)
        self._subs.append(sub)
        return sub

    def _unsubscribe(self, sub: "_Subscription") -> None:
        if sub in self._subs:
            self._subs.remove(sub)
            if self._started and not self._subs:
                self._cancel_pumps()

    def _start(self) -> None:
        """Deliver the backlog to all subscribers and start the source pumps."""
        if self._started:
            return
        self._started = True
        for item in self._backlog:
            for sub in self._subs:
                sub._put(item)
        self._backlog.clear()
        if self._closed:
            self._end()  # closed before the first pull: sources are never started
        else:
            for source in self._pending_sources:
                self._start_pump(source)
        self._pending_sources.clear()

    def _start_pump(self, source: AsyncIterator[Any]) -> None:
        async def pump() -> None:
            try:
                async for item in source:
                    self._put(item)
                    await asyncio.sleep(0)  # let consumers run between items
            except Exception as e:
                if self._error is None:
                    self._error = e
            finally:
                self._open_sources -= 1
                if self._open_sources <= 0:
                    self._end()

        self._pumps.append(asyncio.ensure_future(pump()))

    def _cancel_pumps(self) -> None:
        for task in self._pumps:
            task.cancel()

    def _end(self) -> None:
        """Mark the stream exhausted for all subscribers."""
        self._cancel_pumps()
        for sub in self._subs:
            if not sub._eos:  # idempotent: one _WAKE per subscriber
                sub._eos = True
                sub._queue.put_nowait(_WAKE)


class _Subscription:
    """One consumer's view of a :class:`Stream`: its own queue and windowing."""

    def __init__(self, stream: Stream) -> None:
        self._stream = stream
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._eos = False
        self._woken = False

    def _put(self, item: Any) -> None:
        """Enqueue an item; when conflating, drop the oldest backlog first so the queue stays
        bounded to roughly ``batch_size`` items even while an iteration is running."""
        if self._stream.conflate:
            while self._queue.qsize() >= self._stream.batch_size:
                self._queue.get_nowait()
        self._queue.put_nowait(item)

    def wake(self) -> None:
        """Unblock a waiting collector; subsequent windows are empty."""
        if not self._woken:
            self._woken = True
            self._queue.put_nowait(_WAKE)

    def unsubscribe(self) -> None:
        """Detach from the stream (idempotent)."""
        self._stream._unsubscribe(self)

    async def next_window(self) -> list[Any]:
        """Collect the next window; empty means stopped or exhausted.

        Raises:
            Exception: The error of a failed source, once its items are consumed.
        """
        self._stream._start()
        if self._stream._error is not None:
            raise self._stream._error
        batch_size = self._stream.batch_size
        window: list[Any] = []
        while len(window) < batch_size and not self._woken:
            if self._eos and self._queue.empty():
                break
            if window and self._stream.max_wait is not None:
                item = await self._get_or_timeout(self._stream.max_wait)
                if item is _TIMEOUT:
                    break  # flush partial window
            else:
                item = await self._queue.get()
            if item is not _WAKE:
                window.append(item)

        if self._woken:
            return []

        if self._stream.conflate:
            while not self._queue.empty():
                item = self._queue.get_nowait()
                if item is not _WAKE:
                    window.append(item)
            window = window[-batch_size:]

        if not window and self._stream._error is not None:
            raise self._stream._error
        return window

    async def _get_or_timeout(self, timeout: float) -> Any:
        """Queue get with timeout; unlike wait_for on Python < 3.12, never drops a fetched item."""
        getter = asyncio.ensure_future(self._queue.get())
        try:
            await asyncio.wait({getter}, timeout=timeout)
        except asyncio.CancelledError:
            getter.cancel()
            raise
        if getter.done():
            return getter.result()
        getter.cancel()  # an item fetched before the cancel takes effect stays in the queue
        try:
            return await getter
        except asyncio.CancelledError:
            if getter.cancelled():
                return _TIMEOUT
            raise
