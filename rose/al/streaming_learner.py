import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from typing import Any

from radical.asyncflow import WorkflowEngine

from ..learner import IterationState, Learner, LearnerConfig


class StreamingActiveLearner(Learner):
    """Active learner driven by streamed data instead of a simulation task.

    Data items arrive via :meth:`feed` or :meth:`attach_source` and are
    collected into windows of ``batch_size`` items (a partial window is
    flushed after ``max_wait`` seconds). Each window triggers one learning
    iteration: training -> active learning -> criterion. The window is
    passed as the first positional argument to the training task, and the
    previous iteration's active-learn result is appended as a dependency
    (for warm starts).

    Unlike SequentialActiveLearner, a met stop criterion does not end the
    loop: it marks the model as publishable (``state.should_stop`` is True
    and ``on_model_ready`` callbacks fire) while consumption continues.
    The loop ends when :meth:`stop` is called or all attached sources are
    exhausted.

    Example::

        learner = StreamingActiveLearner(asyncflow, batch_size=10)
        learner.attach_source(sensor_stream())

        async for state in learner.start():
            if state.should_stop:
                publish(state)
    """

    _END = object()  # sentinel: an attached source finished
    _WAKE = object()  # sentinel: stop() unblocking the collector

    def __init__(
        self,
        asyncflow: WorkflowEngine,
        batch_size: int = 1,
        max_wait: float | None = None,
        conflate: bool = False,
        sources: AsyncIterator[Any] | list[AsyncIterator[Any]] | None = None,
    ) -> None:
        """Initialize the Streaming Active Learner.

        Args:
            asyncflow: The workflow engine instance used to manage async tasks.
            batch_size: Number of streamed items per learning window.
            max_wait: Flush a partial window after this many seconds of
                waiting for more items. None waits for a full window.
            conflate: If True, drop the oldest backlog at ingestion time and
                keep only the newest ~``batch_size`` items when iterations
                are slower than the stream ("latest wins"); the internal
                queue stays bounded.
            sources: One or more async iterators to consume as data
                sources; equivalent to calling :meth:`attach_source` for
                each.
        """
        super().__init__(asyncflow, register_and_submit=True)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_wait is not None and max_wait <= 0:
            raise ValueError(f"max_wait must be positive or None, got {max_wait}")
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.conflate = conflate

        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._open_sources = 0
        self._exhausted = False
        self._started = False
        self._sources: list[asyncio.Task] = []
        self._pending_sources: list[AsyncIterator[Any]] = []
        self._model_callbacks: list[Callable[[IterationState], Any]] = []
        self._pending_config: LearnerConfig | None = None

        if sources is not None:
            for source in sources if isinstance(sources, list) else [sources]:
                self.attach_source(source)

    async def feed(self, item: Any) -> None:
        """Feed a single data item into the learner's stream."""
        await self._put(item)

    async def _put(self, item: Any) -> None:
        """Enqueue a data item; when conflating, drop the oldest backlog first so the queue stays
        bounded to roughly ``batch_size`` items even while an iteration is running."""
        if self.conflate:
            while self._queue.qsize() >= self.batch_size:
                self._drain_sentinel(self._queue.get_nowait())
        await self._queue.put(item)

    def attach_source(self, source: AsyncIterator[Any]) -> None:
        """Attach an async iterator as a data source.

        Sources attached before :meth:`start` are only consumed once the learner loop runs. The loop
        ends once all attached sources are exhausted and the queue is drained; learners fed only via
        :meth:`feed` run until :meth:`stop` is called.
        """
        self._open_sources += 1
        if self._started:
            self._start_pump(source)
        else:
            self._pending_sources.append(source)

    def _start_pump(self, source: AsyncIterator[Any]) -> None:
        async def pump() -> None:
            try:
                async for item in source:
                    await self._put(item)
            finally:
                await self._queue.put(self._END)

        self._sources.append(asyncio.ensure_future(pump()))

    def on_model_ready(self, callback: Callable[[IterationState], Any]) -> None:
        """Register a callback fired whenever the stop criterion is met.

        In streaming mode the criterion acts as a publish gate, not a terminal condition. Callbacks
        receive the IterationState and may be sync or async.
        """
        self._model_callbacks.append(callback)

    def set_next_config(self, config: LearnerConfig) -> None:
        """Set configuration to apply from the next window on."""
        self._pending_config = config

    def stop(self) -> None:
        """Signal the learner to stop and unblock the window collector."""
        super().stop()
        self._queue.put_nowait(self._WAKE)

    def _drain_sentinel(self, item: Any) -> bool:
        """Process a sentinel item; return True if it was one."""
        if item is self._WAKE:
            return True
        if item is self._END:
            self._open_sources -= 1
            if self._open_sources <= 0:
                self._exhausted = True
            return True
        return False

    async def _collect(self) -> list[Any]:
        """Collect the next window; empty means stopped or exhausted."""
        window: list[Any] = []
        while len(window) < self.batch_size:
            if self.is_stopped or (self._exhausted and self._queue.empty()):
                break
            try:
                timeout = self.max_wait if window else None
                item = await asyncio.wait_for(self._queue.get(), timeout)
            except asyncio.TimeoutError:
                break  # flush partial window
            if not self._drain_sentinel(item):
                window.append(item)

        if self.conflate:
            while not self._queue.empty():
                item = self._queue.get_nowait()
                if not self._drain_sentinel(item):
                    window.append(item)
            window = window[-self.batch_size :]

        return window

    async def start(
        self, initial_config: LearnerConfig | None = None
    ) -> AsyncIterator[IterationState]:
        """Consume the stream and yield an IterationState per window.

        Args:
            initial_config: Optional LearnerConfig; can be replaced between
                windows via set_next_config().

        Yields:
            IterationState per processed window, with ``window_size`` in
            its state dict. ``should_stop`` marks criterion-met (model
            ready) states; the loop itself keeps running.
        """
        if not self.training_function or not self.active_learn_function:
            raise ValueError("Training and Active Learning functions must be set!")

        self._started = True
        for source in self._pending_sources:
            self._start_pump(source)
        self._pending_sources.clear()

        config = initial_config
        acl_task: Any = None
        _stop_reason = "stream_exhausted"

        try:
            i = 0
            while True:
                window = await self._collect()
                if self.is_stopped:
                    _stop_reason = "stopped"
                    break
                if not window:
                    break  # sources exhausted

                if self._pending_config is not None:
                    config = self._pending_config
                    self._pending_config = None

                self.clear_state()

                train_cfg = self._get_iteration_task_config(
                    self.training_function, config, "training", i
                )
                train_cfg["args"] = (window, *train_cfg["args"])
                train_task = self._register_task(train_cfg, deps=acl_task)
                train_result = await train_task

                acl_cfg = self._get_iteration_task_config(
                    self.active_learn_function, config, "active_learn", i
                )
                acl_task = self._register_task(acl_cfg, deps=train_task)
                acl_result = await acl_task

                if self.is_stopped:
                    _stop_reason = "stopped"
                    break
                self._extract_state_from_result(train_result)
                self._extract_state_from_result(acl_result)

                metric_value: float | None = None
                should_stop = False
                if self.criterion_function:
                    crit_cfg = self._get_iteration_task_config(
                        self.criterion_function, config, "criterion", i
                    )
                    stop_result = await self._register_task(crit_cfg)
                    if self.is_stopped:
                        _stop_reason = "stopped"
                        break
                    should_stop, metric_value = self._check_stop_criterion(stop_result)

                self.register_state("window_size", len(window))
                self._iteration_state = self.build_iteration_state(
                    iteration=i,
                    metric_value=metric_value,
                    should_stop=should_stop,
                    current_config=config,
                )

                self._notify_trackers_iteration(self._iteration_state)
                if should_stop:
                    for cb in self._model_callbacks:
                        result = cb(self._iteration_state)
                        if inspect.isawaitable(result):
                            await result

                yield self._iteration_state
                i += 1
        except Exception:
            _stop_reason = "error"
            raise
        finally:
            self._notify_trackers_stop(self._iteration_state, _stop_reason)
            for task in self._sources:
                task.cancel()
