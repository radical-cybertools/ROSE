"""Task functions for the streaming spec example: fit y = a*x + b on all readings seen so far."""


def simulate(window, *args, **kwargs):
    # later windows also receive the previous active-learning result (all readings so far)
    seen = args[0]["readings"] if args else []
    return {"readings": seen + list(window)}


def train(sim_result, **kwargs):
    pts = sim_result["readings"]
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    var = sum((x - mx) ** 2 for x, _ in pts) or 1e-12
    a = sum((x - mx) * (y - my) for x, y in pts) / var
    return {"a": a, "b": my - a * mx}


def active_learn(sim_result, model, **kwargs):
    pts = sim_result["readings"]
    mse = sum((model["a"] * x + model["b"] - y) ** 2 for x, y in pts) / len(pts)
    # dict results become IterationState fields: state.mse, state.model in on_model_ready
    return {"readings": pts, "model": model, "mse": mse}
