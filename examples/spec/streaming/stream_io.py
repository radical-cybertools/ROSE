"""Client-side stream source and publish callback for the streaming spec example."""

import asyncio
import random


async def sensor(n: int = 12, delay: float = 0.1):
    """Simulated sensor: yields n (x, y) readings of y = 2x + 1 plus noise."""
    for _ in range(n):
        x = random.random()
        yield (x, 2 * x + 1 + random.gauss(0, 0.05))
        await asyncio.sleep(delay)


def publish(state):
    print(f"publish model {state.model} from iteration {state.iteration} (mse={state.mse:.4f})")
