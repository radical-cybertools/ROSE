"""Toy task functions for the UQ spec example: two models fit y = 2x + 1 from noisy samples.

UQ adds dash-named kwargs ("--model_name", "--iteration"), so every function takes **kwargs.
"""

import random


def simulate(*args, **kwargs):
    xs = [random.random() for _ in range(20)]
    return [(x, 2 * x + 1 + random.gauss(0, 0.05)) for x in xs]


def train(data, **kwargs):
    n = len(data)
    mx = sum(x for x, _ in data) / n
    my = sum(y for _, y in data) / n
    a = sum((x - mx) * (y - my) for x, y in data) / sum((x - mx) ** 2 for x, _ in data)
    if kwargs["--model_name"] == "poly":
        a *= 1.01  # a slightly different second model
    return {"a": a, "b": my - a * mx}


def predict(model, **kwargs):
    return [model["a"] * x + model["b"] for x in (0.25, 0.5, 0.75)]


def ensemble_variance(per_model_predictions, **kwargs):
    # one argument: per model, a list of its predictions; variance per point across all
    preds = [p for model_preds in per_model_predictions for p in model_preds]
    points = list(zip(*preds, strict=True))
    return sum(sum((v - sum(pt) / len(pt)) ** 2 for v in pt) / len(pt) for pt in points)


def active_learn(uq_value, **kwargs):
    return {"uq": uq_value}


def check_mse(acl_result, **kwargs):
    return acl_result["uq"]
