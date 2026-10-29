"""Lightweight task helpers used by the YAML spec integration tests."""

# Shared capture list — cleared by tests that need to inspect received kwargs.
received_kwargs: list[dict] = []


async def sim(*args, **kwargs):
    return [1.0, 2.0, 3.0]


async def train(data, **kwargs):
    return {"mean": sum(data) / len(data)}


async def active_learn(sim_result, model, **kwargs):
    return abs(model["mean"] - 2.0)


async def criterion(*args, **kwargs):
    return 0.05


# ── Parameter-capturing variants ──────────────────────────────────────────────


async def sim_capture(*args, **kwargs):
    received_kwargs.append(dict(kwargs))
    return [1.0, 2.0, 3.0]


async def train_capture(data, **kwargs):
    return {"mean": sum(data) / len(data)}


async def active_learn_capture(sim_result, model, **kwargs):
    return abs(model["mean"] - 2.0)


async def criterion_capture(*args, **kwargs):
    return 0.05


# ── Stream helpers ────────────────────────────────────────────────────────────

received_windows: list = []
published: list = []


async def numbers(n=4):
    for i in range(n):
        yield i


async def sim_window(window, *args, **kwargs):
    received_windows.append(list(window))
    return [1.0, 2.0, 3.0]


def publish(state):
    published.append(state.iteration)


async def env_window(window, *args, **kwargs):
    received_windows.append(list(window))
    return {"reward": 1.0}


async def update(*args, **kwargs):
    return {"loss": 0.1}


# ── UQ helpers ────────────────────────────────────────────────────────────────

uq_calls: list = []


async def uq_sim(*args, **kwargs):
    uq_calls.append(("sim", list(args[0]) if args else None, dict(kwargs)))
    return [1.0, 2.0, 3.0]


async def uq_train(sim_result, **kwargs):
    uq_calls.append(("train", kwargs.get("--model_name"), dict(kwargs)))
    return {"model": kwargs.get("--model_name")}


async def uq_predict(model, **kwargs):
    return [0.1, 0.2]


async def uq_uncertainty(*per_model_preds, **kwargs):
    return 0.3


async def uq_active_learn(uq_value, **kwargs):
    return {"selected": 1}


async def uq_criterion(acl_result, **kwargs):
    uq_calls.append(("crit", kwargs.get("--model_name"), dict(kwargs)))
    return 0.5
