import jax


def register_pytree_namedtuple(cls: object):
    jax.tree_util.register_pytree_node(
        cls, lambda xs: (tuple(xs), None), lambda _, xs: cls(*xs)
    )


def delete_small_numbers(arr: jax.numpy.ndarray) -> jax.numpy.ndarray:
    return jax.numpy.where(abs(arr) <= 1e-200, 0.0, arr)
