import itertools
import jax.numpy as jnp
import numpy as np

import scipy
import jax.experimental.sparse as jsparse

from .hilbert import PBCBox1D
from .utils import sorted_multi_particle_state


def coo_kinetic_term(
    hilbert: PBCBox1D,
    hbar: float,
    k_r: float,
    m_Yb: float,
    delta: float,
    omega_R: float
):
    data = []
    rows = []
    cols = []
    for state_index, multi_particle_state in enumerate(hilbert.multi_particle_states):
        i = 0
        N = hilbert.n_particles
        diag_val = 0.0
        while i < N:
            momentum_idx, spin = multi_particle_state[i]
            q_x = hilbert.momentums[momentum_idx]
            if (i + 1 < N) and (momentum_idx == multi_particle_state[i + 1][0]):
                # if state contains c_up(k) and c_down(k) both
                diag_val += ((hbar ** 2) / (2 * m_Yb)) * ((q_x - k_r) ** 2 + (q_x + k_r) ** 2)
                i += 2
            else:
                # state contains either c_up(k) or c_down(k), but not both.
                spin_changed = list(multi_particle_state)   # copy state
                spin_changed[i] = (momentum_idx, -spin)     # reflect spin
                spin_changed_tuple = tuple(spin_changed)
                spin_changed_index = hilbert.state_to_index_map_multi[spin_changed_tuple]
                
                # diagonal term
                diag_val += ((hbar ** 2) / (2 * m_Yb)) * ((q_x - spin * k_r) ** 2) + spin * delta / 2
                
                # off-diagonal term
                val = omega_R / 2
                data.append(val)
                rows.append(spin_changed_index)
                cols.append(state_index)
                i += 1
                
        data.append(diag_val)
        rows.append(state_index)
        cols.append(state_index)
                
    return data, rows, cols


def coo_interaction_term(
    hilbert: PBCBox1D,
    U: float
):
    data = []
    rows = []
    cols = []
    for state_index, multi_particle_state in enumerate(hilbert.multi_particle_states):
        spin_up_states = [(idx, single[0]) for idx, single in enumerate(multi_particle_state) if single[1] == 1]
        spin_down_states = [(idx, single[0]) for idx, single in enumerate(multi_particle_state) if single[1] == -1]
        
        for spin_up_state, spin_down_state in itertools.product(spin_up_states, spin_down_states):
            idx1, momentum_1_idx_i = spin_up_state
            idx2, momentum_2_idx_i = spin_down_state

            for q_idx in hilbert.orbital_indices:
                k1 = momentum_1_idx_i + hilbert.n_min
                k2 = momentum_2_idx_i + hilbert.n_min
                q = q_idx + hilbert.n_min
                
                if k1 + q < hilbert.n_min or k1 + q > hilbert.n_max or k2 - q < hilbert.n_min or k2 - q > hilbert.n_max:
                    continue
                momentum_1_idx_f = k1 + q - hilbert.n_min
                momentum_2_idx_f = k2 - q - hilbert.n_min
                    
                new_multi_particle_state = list(multi_particle_state)
                new_multi_particle_state[idx1] = (momentum_1_idx_f, 1)
                new_multi_particle_state[idx2] = (momentum_2_idx_f, -1)
                new_multi_particle_state, parity = sorted_multi_particle_state(new_multi_particle_state)
                if parity == 0:
                    continue
                new_idx = hilbert.state_to_index_map_multi[new_multi_particle_state]

                data.append(parity * U / hilbert.L)
                rows.append(new_idx)
                cols.append(state_index)
                
    return data, rows, cols
    

def dense_hamiltonian(
    hilbert: PBCBox1D,
    hbar: float,
    k_r: float,
    m_Yb: float,
    delta: float,
    omega_R: float,
    U: float,
    kinetic: bool = True,
    interaction: bool = True
):
    hamiltonian = np.zeros((hilbert.space_dim, hilbert.space_dim), np.complex128)
    
    if kinetic:
        data, rows, cols = coo_kinetic_term(
            hilbert,
            hbar,
            k_r,
            m_Yb,
            delta,
            omega_R
        )
        for val, row, col in zip(data, rows, cols):
            hamiltonian[row, col] += val

    if interaction and np.abs(U) > 1e-10:
        data, rows, cols = coo_interaction_term(
            hilbert,
            U
        )
        for val, row, col in zip(data, rows, cols):
            hamiltonian[row, col] += val
            
    return hamiltonian


def sparse_hamiltonian_scipy_csr(
    hilbert: PBCBox1D,
    hbar: float,
    k_r: float,
    m_Yb: float,
    delta: float,
    omega_R: float,
    U: float,
    kinetic: bool = True,
    interaction: bool = True
):
    data = []
    rows = []
    cols = [] 
    
    if kinetic:
        _v, _r, _c = coo_kinetic_term(
            hilbert,
            hbar,
            k_r,
            m_Yb,
            delta,
            omega_R
        )
        data += _v
        rows += _r
        cols += _c

    if interaction and np.abs(U) > 1e-10:
        _v, _r, _c = coo_interaction_term(
            hilbert,
            U
        )
        data += _v
        rows += _r
        cols += _c
        
    hamiltonian = scipy.sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(hilbert.space_dim, hilbert.space_dim),
        dtype=np.complex128
    )
    return hamiltonian.tocsr()

def sparse_hamiltonian_jax_bcsr(
    hilbert: PBCBox1D,
    hbar: float,
    k_r: float,
    m_Yb: float,
    delta: float,
    omega_R: float,
    U: float,
    kinetic: bool = True,
    interaction: bool = True
) -> jsparse.JAXSparse:
    data = []
    rows = []
    cols = [] 
    
    if kinetic:
        _v, _r, _c = coo_kinetic_term(
            hilbert,
            hbar,
            k_r,
            m_Yb,
            delta,
            omega_R
        )
        data += _v
        rows += _r
        cols += _c

    if interaction and np.abs(U) > 1e-10:
        _v, _r, _c = coo_interaction_term(
            hilbert,
            U
        )
        data += _v
        rows += _r
        cols += _c
    
    data = jnp.array(data, dtype=jnp.complex128)
    rows = jnp.array(rows, dtype=jnp.int32)
    cols = jnp.array(cols, dtype=jnp.int32)
    indices = jnp.concat([rows, cols], axis=-1)

    hamiltonian_bcoo = jsparse.BCOO(
        (data, indices),
        shape=(hilbert.space_dim, hilbert.space_dim),
    )
    return jsparse.BCSR.from_bcoo(hamiltonian_bcoo)