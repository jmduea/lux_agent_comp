# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
import jax.numpy as jnp


def direction_to(src, target):
    if not isinstance(src, jnp.ndarray):
        src = jnp.array(src)
    if not isinstance(target, jnp.ndarray):
        target = jnp.array(target)
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1
