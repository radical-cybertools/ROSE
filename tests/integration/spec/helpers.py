"""Lightweight task helpers used by the YAML spec integration tests."""


async def sim(*args, **kwargs):
    return [1.0, 2.0, 3.0]


async def train(data, **kwargs):
    return {"mean": sum(data) / len(data)}


async def active_learn(sim_result, model, **kwargs):
    return abs(model["mean"] - 2.0)


async def criterion(*args, **kwargs):
    return 0.05
