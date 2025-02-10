

import jax.numpy as jnp


from . import multi_index






def position_arr(lengths, lattice_constant):
    sites_multi_indices = multi_index.multi_indices(lengths)
    position_arr = lattice_constant * jnp.array(sites_multi_indices, dtype=jnp.float64)
    return list(position_arr)


def momentum_arr(lengths, lattice_constant):
    sites_multi_indices = multi_index.multi_indices(lengths)
    site_index_arr = jnp.array(sites_multi_indices, dtype=jnp.float64)
    
    k0 = 2 * jnp.pi / (lattice_constant * jnp.array(lengths))
    momentum_arr = site_index_arr * k0
    momentum_arr = jnp.where(momentum_arr > jnp.pi / lattice_constant, 
                             momentum_arr - 2 * jnp.pi / lattice_constant, 
                             momentum_arr)
    return list(momentum_arr)