import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

from jaxtyping import PRNGKeyArray


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

    Returns:
        est (float): An underestimate of the 1-norm of the sparse matrix.
    """
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    
    n = A.shape[1]
    if t >= n:
        A_explicit = jnp.asarray()