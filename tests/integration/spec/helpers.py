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
