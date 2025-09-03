from math import ceil
from functools import partial
import scipy
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse


def expm_steps_est(op: scipy.sparse.linalg.LinearOperator, norm_bound=1.0):
    one_norm = scipy.sparse.linalg.onenormest(op)
    return ceil(one_norm / norm_bound)


@partial(jax.jit, static_argnums=(2,))
def expm_taylor(
    A: jsparse.JAXSparse,
    B: jax.Array,
    order: int
):
    """calculate and apply the matrix exponential

    compute $e^A B$ using a Taylor expansion of the matrix exponential.

    Args:
        A (jax.experimental.sparse.JAXSparse): Sparse matrix to calculate the matrix exponential.
        B (jax.Array): dense matrix to multiply the matrix exponential with.
        order (int): order of the Taylor expansion.
    """
    if order == 0:
        return B

    init_state = (B, B)

    def taylor_body(i, state):
        acc, carry = state
        carry = A @ carry / i
        acc = acc + carry
        return acc, carry

    final_val, _ = jax.lax.fori_loop(1, order + 1, taylor_body, init_state)
    return final_val

@partial(jax.jit, static_argnums=(2, 3))
def apply_expm(
    A_scaled: jsparse.JAXSparse,
    B: jax.Array,
    n_steps: int,
    taylor_order: int
):
    """apply the matrix exponential to a state
    compute $e^{A} B = (e^{A/n})^n B$

    Args:
        A (jax.experimental.sparse.JAXSparse): Sparse matrix to calculate the matrix exponential.
        B (jax.Array): dense matrix to multiply the matrix exponential with.
        n_steps (int): number of steps to apply the taylor polynomials.
        taylor_order (int): order of the Taylor expansion.
    """
    def step_body(i, state):
        return expm_taylor(A_scaled, state, taylor_order)

    exp_A_B = jax.lax.fori_loop(0, n_steps, step_body, B)
    return exp_A_B