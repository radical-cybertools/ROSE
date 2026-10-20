# Streaming Active Learning

A `Stream` drives an existing active learner with streamed data (sensor windows,
events) instead of an internal iteration count. This fits digital-twin runtimes,
where data arriving after startup should keep improving the model.

```python
from rose import Stream
from rose.al.active_learner import SequentialActiveLearner

stream = Stream(batch_size=10, max_wait=2.0)
stream.attach_source(sensor_stream())    # any async iterator
# or, from elsewhere: await stream.feed(item)

learner = SequentialActiveLearner(asyncflow)
learner.on_model_ready(publish_model)    # sync or async callback

async for state in learner.start(stream=stream):
    print(state.iteration, state.window_size, state.model_ready)
```

A runnable example lives in `examples/active_learn/streaming/run_me.py`.

## Windows

- `batch_size` items form a window; each window triggers one
  simulation → training → active-learning → criterion iteration.
- `max_wait` flushes a partial window after that many seconds without new items.
- `conflate=True` drops the oldest backlog and keeps the newest items when
  iterations are slower than the stream (latest wins).
- Items fed before the learner starts are kept and delivered to it.

## Task call shapes

The window is the first argument of the first pipeline task. Task results are
appended as dependencies, as in batch mode:

| Task | Call |
|---|---|
| simulation, first window | `simulation(window, *args)` |
| simulation, later windows | `simulation(window, *args, acl_result)` |
| training | `training(*args, sim_result)` |
| active learning | `active_learn(*args, sim_result, train_result)` |

With `skip_simulation_step=True` the window goes to training instead:
`training(window, *args)`, later `training(window, *args, acl_result)`.

Executable tasks receive the window as first argument of their command-builder
function; YAML spec shell tasks ignore it.

## Publishing models

In stream mode a met stop criterion does not end the loop. It sets
`state.model_ready` and fires the `on_model_ready` callbacks; without a criterion
every window's model is ready. `state.should_stop` still reports criterion-met,
so `if state.should_stop: break` leaves the stream loop.

The callback receives the `IterationState`, which carries what tasks registered
(dict results of function tasks). Executable tasks must pass model references
themselves, e.g. a checkpoint path in a known location. Callback exceptions are
logged and ignored.

## Termination

The loop ends when

- `learner.stop()` is called (also unblocks a learner waiting for data),
- the stream is exhausted: all attached sources finished, or `stream.close()`,
- `max_iter` windows were processed (optional cap),
- a source raises: the error is re-raised from `start()`.

`teach(stream=...)` on an unbounded stream returns only after `stop()` or
`stream.close()`. `stream` cannot be combined with `skip_pre_loop=True`.

## Parallel learners

`ParallelActiveLearner.start(stream=...)` delivers every item to every learner.
Windows are formed per learner, so they are identical across learners only with
`max_wait=None` and `conflate=False`. `on_model_ready` callbacks registered on the
parallel learner receive states with `learner_id` set. A failing learner stops
all others. To share one stream between learners, use `ParallelActiveLearner`.
