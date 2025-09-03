from functools import partial

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ..jax_linalg.expm_utils import apply_expm

from einops import rearrange, pack


@partial(jax.jit, static_argnames=('n_expm_steps', 'taylor_order', 'n_loss_channels', 'loss_op_rank'), donate_argnames=('psi_batched',))
def qjm_step(
    psi_batched: Complex[Array, "hdim batch"],
    scaled_exponent: jsparse.JAXSparse,
    stacked_loss_op: jsparse.JAXSparse,
    delta_t: Float[Array, ""],
    hbar: Float[Array, ""],
    n_expm_steps: int,
    taylor_order: int,
    n_loss_channels: int,
    loss_op_rank: int,
    T2: Float[Array, ""],
    *,
    key: PRNGKeyArray,
):
    hilb_dim, batch_size = psi_batched.shape
    subspace_dim = loss_op_rank
    
    # non-hermitian evolution
    psi_evolved = apply_expm(
        scaled_exponent, # (-1j * delta_t / hbar) * H_eff,
        psi_batched,
        n_steps=n_expm_steps,
        taylor_order=taylor_order
    )

    # apply loss operator
    loss_jumped = rearrange(
        stacked_loss_op @ psi_evolved,
        "(channels state) batch -> channels state batch",
        channels=n_loss_channels
    )

    loss_jump_probs = jnp.linalg.norm(loss_jumped, axis=1) ** 2 * delta_t # channels state batch -> channels batch
    dephase_jump_probs = (1 / T2) * (jnp.abs(psi_evolved) ** 2) * delta_t # (hdim, batch)
    no_jump_probs = 1 - (jnp.sum(loss_jump_probs, axis=0, keepdims=True) + jnp.sum(dephase_jump_probs, axis=0, keepdims=True))

    jump_probs, _ = pack(
        [loss_jump_probs, dephase_jump_probs, no_jump_probs],
        "* batch"
    )

    # determine jump channel
    key, subkey = jax.random.split(key)
    cum_p = jnp.cumsum(jump_probs, axis=0)                       # (C_total, B)
    rand = jax.random.uniform(subkey, (batch_size,))             # (B,)
    jump_indices = jax.vmap(
        lambda x, y: jnp.searchsorted(x, y, side="right"),
        in_axes=(1, 0)
    )(cum_p, rand)                                               # (B,)

    # apply jump
    def _apply_jump(psi, loss_jumped, jump_index):
        branch = jnp.where(jump_index < n_loss_channels,            0,
                 jnp.where(jump_index < n_loss_channels + hilb_dim, 1,
                                                                    2))
        def _loss(_):
            new = jnp.zeros_like(psi)
            new = new.at[:subspace_dim].set(loss_jumped[jump_index, :])
            norm = jnp.linalg.norm(new)
            return new / norm
        def _dephase(_):
            return jax.nn.one_hot(jump_index - n_loss_channels, hilb_dim, dtype=jnp.complex128)
        def _no_jump(_):
            norm = jnp.linalg.norm(psi)
            return psi / norm

        psi_new = jax.lax.switch(branch, [_loss, _dephase, _no_jump], None)
        return psi_new

    psi_jumped = jax.vmap(_apply_jump, in_axes=(1, 2, 0), out_axes=1)(
        psi_evolved,
        loss_jumped,
        jump_indices
    )
    return psi_jumped