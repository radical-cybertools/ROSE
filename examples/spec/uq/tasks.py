"""Toy task functions for the UQ spec example: two models fit y = 2x + 1 from noisy samples.

The "poly" model is biased early and converges with the iteration, so the ensemble variance
shrinks until the uncertainty criterion ends the run. UQ adds dash-named kwargs
("--model_name", "--iteration"), so every function takes **kwargs.
"""

import random


def simulate(*args, **kwargs):
    xs = [random.random() for _ in range(50)]
    return [(x, 2 * x + 1 + random.gauss(0, kwargs["noise"])) for x in xs]


def train(data, **kwargs):
    n = len(data)
    mx = sum(x for x, _ in data) / n
    my = sum(y for _, y in data) / n
    a = sum((x - mx) * (y - my) for x, y in data) / sum((x - mx) ** 2 for x, _ in data)
    if kwargs["--model_name"] == "poly":
        a *= 1 + 0.2 / (kwargs["iteration"] + 1)  # biased early, converging later
    return {"a": a, "b": my - a * mx}


def predict(model, **kwargs):
    return [model["a"] * x + model["b"] for x in (0.25, 0.5, 0.75)]


def ensemble_variance(per_model_predictions, **kwargs):
    # one argument: per model, a list of its predictions; variance per point across all
    preds = [p for model_preds in per_model_predictions for p in model_preds]
    points = list(zip(*preds, strict=True))
    return sum(sum((v - sum(pt) / len(pt)) ** 2 for v in pt) / len(pt) for pt in points)


def active_learn(uq_value, **kwargs):
    print(f"iteration {kwargs['iteration']}: ensemble variance {uq_value:.5f}")
    return {"variance": uq_value}


def slope_error(acl_result, **kwargs):
    return 1.0  # placeholder per-model check; the uncertainty criterion ends this run
