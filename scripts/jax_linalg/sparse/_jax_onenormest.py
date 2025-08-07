from functools import partial

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

from jaxtyping import PRNGKeyArray

@jax.jit
def sign_round_up(x: jax.Array):
    return jnp.where(x == 0, 1.0, jnp.sign(x))



@jax.jit
def every_col_of_X_is_parallel_to_a_col_of_Y(X, Y):
    """X의 모든 열 벡터가 Y의 열 벡터 중 하나와 평행한지 확인합니다."""
    def is_parallel_to_any_in_Y(v, Y):
        # v가 Y의 열들과 평행한지 여부를 vmap으로 계산
        return jnp.any(jax.vmap(vectors_are_parallel, in_axes=(None, 1))(v, Y))

    # X의 모든 열에 대해 vmap 적용
    return jnp.all(jax.vmap(is_parallel_to_any_in_Y, in_axes=(1, None))(X, Y))
    
    
    
    
    
    
    

def _onenormest_core_jax(
    A: jsparse.JAXSparse, 
    AT: jsparse.JAXSparse, 
    t: int, 
    itmax: int,
    key: PRNGKeyArray
):
    """Compute a lower bound of the 1-norm of a sparse array.

    Args:
        A (jax.experimental.sparse.JAXSparse): An square sparse matrix.
        AT (jax.experimental.sparse.JAXSparse): The transpose of A.
        t (int): A parameter controlling the 
        tradeoff between accuracy versus time and memory usage. At least 1 and must less than size of A
        itmax (int): Use at most this many iterations. At least 2.
        key (PRNGKeyArray): jax PRNG key
    """
    
    n = A.shape[0]
    
    if t == 1:
        X = jnp.ones((n, 1))
    elif t > 1:
        X1 = jnp.ones((n, 1))
        key, subkey = jax.random.split(key)
        X2 = jax.random.choice(subkey, [1, -1], shape=(n, t-1))
        signs = jax.random.choice(subkey, [1, -1], shape=(1, t-1))
        X2 = X2 * signs
        X = jnp.concatenate([X1, X2], axis=1)
    else:
        raise ValueError(f"t(={t}) < 1 is not allowed.")
    
    X = X / n
    
    # resampling?
    # (k, est, est_old, w, ind_best, X, S, S_old, ind, ind_hist, n_mults, key)
    init_val = (
        1,
        0,
        0,
        jnp.zeros(n),
        -1,
        X,
        jnp.zeros((n, t)),
        jnp.zeros((n, t)),
        jnp.zeros(t, dtype=jnp.int32),
        jnp.array([], dtype=jnp.int32),
        0,
        key
    )
    
    def cond_fun(val):
        k, est, est_old, _, _, _, S, S_old, _, _, _, _ = val
        stop_1 = (k >= 2) & (est <= est_old)
        stop_2 = 
    
    
    



def onenormest(
    A: jsparse.JAXSparse, 
    t: int=2, 
    itmax: int=5, 
    *, 
    key: PRNGKeyArray
):
    """
    Compute a lower bound of the 1-norm of a sparse matrix A using randomized algorithms.

    Args:
        A (jax.experimental.sparse.JAXSparse): The input sparse matrix.
        t (int, optional): A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
        Larger values take longer and use more memory but give more accurate output.
        itmax (int, optional): Use at most this many iterations.
        key (PRNGKeyArray): A JAX random key for reproducibility.
    Returns:
        est (float): An underestimate of the 1-norm of the sparse matrix.
    """
    
    n = A.shape[0]
    
    key, subkey = jax.random.split(key)
    X = jax.random.choice(subkey, jnp.array([-1.0, 1.0]), shape=(n, t))