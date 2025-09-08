import itertools
import math

import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np
import scipy

from .hilbert import PBCBox1D
from .utils import sorted_multi_particle_state


def trace_e(q_x, m_Yb, k_r, delta, omega_R, hbar=1.0):
    return ((hbar ** 2) / (2 * m_Yb)) * (
        (q_x + k_r) ** 2 + (q_x - k_r) ** 2
    )

def delta_e(q_x, m_Yb, k_r, delta, omega_R, hbar=1.0):
    return ((hbar ** 2) / (2 * m_Yb)) * (
        (q_x + k_r) ** 2 - (q_x - k_r) ** 2
    ) - delta

def e1(q_x, m_Yb, k_r, delta, omega_R, hbar=1.0):
    tr = trace_e(q_x, m_Yb, k_r, delta, omega_R, hbar)
    de = delta_e(q_x, m_Yb, k_r, delta, omega_R, hbar)
    return (tr + jnp.sqrt(de ** 2 + omega_R**2)) / 2

def e2(q_x, m_Yb, k_r, delta, omega_R, hbar=1.0):
    tr = trace_e(q_x, m_Yb, k_r, delta, omega_R, hbar)
    de = delta_e(q_x, m_Yb, k_r, delta, omega_R, hbar)
    return (tr - jnp.sqrt(de ** 2 + omega_R**2)) / 2

def theta(q_x, m_Yb, k_r, delta, omega_R, hbar=1.0):
    de = delta_e(q_x, m_Yb, k_r, delta, omega_R, hbar)
    assert omega_R >= 0
    
    if omega_R == 0.0:
        return 0.0
    
    if -1e-12 < de <= 0:
        return -np.pi / 4
    elif 0 <= de < 1e-12:
        return np.pi / 4
        
    return (1 / 2) * jnp.arctan(omega_R / de)

def alpha(q_x, m_Yb, k_r, delta, omega_R, hbar=1.0):
    return jnp.cos(theta(q_x, m_Yb, k_r, delta, omega_R, hbar))

def beta(q_x, m_Yb, k_r, delta, omega_R, hbar=1.0):
    return jnp.sin(theta(q_x, m_Yb, k_r, delta, omega_R, hbar))

def soc_spectrum(
    hilb: PBCBox1D, hbar: float, k_r: float, m_Yb: float, delta: float, omega_R: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(hilb.n_particles, int):
        raise ValueError(f"Hilber space must be an sector that particle number is fixed. but {hilb.n_particles=}")
    
    single_spectrum = []
    single_states = []
    full_momentum_modes = []
    
    for mode_idx, q_x in enumerate(hilb.momentums):
        _e1 = e1(q_x, m_Yb, k_r, delta, omega_R, hbar)
        _e2 = e2(q_x, m_Yb, k_r, delta, omega_R, hbar)
        _alpha = alpha(q_x, m_Yb, k_r, delta, omega_R, hbar)
        _beta = beta(q_x, m_Yb, k_r, delta, omega_R, hbar)
        
        single_spectrum.append(_e1)
        full_momentum_modes.append(mode_idx)
        single_states.append([_alpha, _beta])
        single_spectrum.append(_e2)
        full_momentum_modes.append(mode_idx)
        single_states.append([-_beta, _alpha])
    
    single_spectrum = np.array(single_spectrum)
    full_momentum_modes = np.array(full_momentum_modes)
    single_states = np.array(single_states)

    sorted_idx = np.argsort(single_spectrum)
    single_spectrum = single_spectrum[sorted_idx]
    full_momentum_modes = full_momentum_modes[sorted_idx]
    single_states = single_states[sorted_idx]
    
    return single_spectrum, full_momentum_modes, single_states
    
def single_to_multi_spectrum(
    single_spectrum: np.ndarray, n_particles
) -> tuple[np.ndarray, np.ndarray]:
    n_single_states = single_spectrum.shape[0]
    idx_combinations = np.array(list(itertools.combinations(range(n_single_states), r=n_particles)))
    multi_spectrum = np.array([np.sum(single_spectrum[ii]) for ii in idx_combinations])
    sorted_idx = np.argsort(multi_spectrum)

    multi_spectrum = multi_spectrum[sorted_idx]
    idx_combinations = idx_combinations[sorted_idx]
    
    return multi_spectrum, idx_combinations

def manybody_spectrum(
    hilb: PBCBox1D, hbar: float, k_r: float, m_Yb: float, delta: float, omega_R: float
):
    single_spectrum, full_momentum_modes, single_states = soc_spectrum(hilb, hbar, k_r, m_Yb, delta, omega_R)
    multi_spectrum, idx_combinations = single_to_multi_spectrum(single_spectrum, hilb.n_particles)
    assert single_spectrum.shape[0] == hilb.num_single_particle_states
    assert idx_combinations.shape[0] == hilb.num_multi_particle_states
    
    lowlyings = ([(int(m), alphabeta) for m, alphabeta in zip(full_momentum_modes[ii], single_states[ii])]
                  for ii in idx_combinations)
    
    return multi_spectrum, lowlyings
    

def coo_kinetic_term(
    hilb: PBCBox1D, hbar: float, k_r: float, m_Yb: float, delta: float, omega_R: float
):
    data = []
    rows = []
    cols = []
    for state_index, multi_particle_state in enumerate(hilb.multi_particle_states):
        i = 0
        N = len(multi_particle_state)
        diag_val = 0.0
        while i < N:
            momentum_idx, spin = multi_particle_state[i]
            q_x = hilb.momentums[momentum_idx]
            if (i + 1 < N) and (momentum_idx == multi_particle_state[i + 1][0]):
                # if state contains c_up(k) and c_down(k) both
                diag_val += ((hbar**2) / (2 * m_Yb)) * (
                    (q_x + k_r) ** 2 + (q_x - k_r) ** 2
                )
                i += 2
            else:
                # state contains either c_up(k) or c_down(k), but not both.
                spin_changed = list(multi_particle_state)  # copy state
                spin_changed[i] = (momentum_idx, -spin)  # reflect spin
                spin_changed_tuple = tuple(spin_changed)
                spin_changed_index = hilb.state_to_index_map_multi[spin_changed_tuple]

                # diagonal term
                diag_val += ((hbar**2) / (2 * m_Yb)) * (
                    (q_x + spin * k_r) ** 2
                ) + spin * delta / 2

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


def coo_interaction_term(hilb: PBCBox1D, U: complex):
    data = []
    rows = []
    cols = []
    for state_index, multi_particle_state in enumerate(hilb.multi_particle_states):
        spin_up_states = [
            (idx, single[0])
            for idx, single in enumerate(multi_particle_state)
            if single[1] == 1
        ]
        spin_down_states = [
            (idx, single[0])
            for idx, single in enumerate(multi_particle_state)
            if single[1] == -1
        ]

        for spin_up_state, spin_down_state in itertools.product(
            spin_up_states, spin_down_states
        ):
            idx1, momentum_1_idx_i = spin_up_state
            idx2, momentum_2_idx_i = spin_down_state

            for q_idx in hilb.orbital_indices:
                k1 = momentum_1_idx_i + hilb.n_min
                k2 = momentum_2_idx_i + hilb.n_min
                q = q_idx + hilb.n_min

                if (
                    k1 + q < hilb.n_min
                    or k1 + q > hilb.n_max
                    or k2 - q < hilb.n_min
                    or k2 - q > hilb.n_max
                ):
                    continue
                momentum_1_idx_f = k1 + q - hilb.n_min
                momentum_2_idx_f = k2 - q - hilb.n_min

                new_multi_particle_state = list(multi_particle_state)
                new_multi_particle_state[idx1] = (momentum_1_idx_f, 1)
                new_multi_particle_state[idx2] = (momentum_2_idx_f, -1)
                new_multi_particle_state, parity = sorted_multi_particle_state(
                    new_multi_particle_state
                )
                if parity == 0:
                    continue
                new_idx = hilb.state_to_index_map_multi[new_multi_particle_state]

                data.append(parity * U / hilb.L)
                rows.append(new_idx)
                cols.append(state_index)

    return data, rows, cols


def dense_hamiltonian(
    hilb: PBCBox1D,
    hbar: float,
    k_r: float,
    m_Yb: float,
    delta: float,
    omega_R: float,
    U: complex,
    kinetic: bool = True,
    interaction: bool = True,
):
    hamiltonian = np.zeros((hilb.space_dim, hilb.space_dim), np.complex128)

    if kinetic:
        data, rows, cols = coo_kinetic_term(hilb, hbar, k_r, m_Yb, delta, omega_R)
        for val, row, col in zip(data, rows, cols):
            hamiltonian[row, col] += val

    if interaction and np.abs(U) > 1e-10:
        data, rows, cols = coo_interaction_term(hilb, U)
        for val, row, col in zip(data, rows, cols):
            hamiltonian[row, col] += val

    return hamiltonian


def sparse_hamiltonian_scipy_csr(
    hilb: PBCBox1D,
    hbar: float,
    k_r: float,
    m_Yb: float,
    delta: float,
    omega_R: float,
    U: complex,
    kinetic: bool = True,
    interaction: bool = True,
):
    data = []
    rows = []
    cols = []

    if kinetic:
        _v, _r, _c = coo_kinetic_term(hilb, hbar, k_r, m_Yb, delta, omega_R)
        data += _v
        rows += _r
        cols += _c

    if interaction and np.abs(U) > 1e-10:
        _v, _r, _c = coo_interaction_term(hilb, U)
        data += _v
        rows += _r
        cols += _c

    hamiltonian = scipy.sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(hilb.space_dim, hilb.space_dim),
        dtype=np.complex128,
    )
    return hamiltonian.tocsr()


def sparse_hamiltonian_jax_bcsr(
    hilb: PBCBox1D,
    hbar: float,
    k_r: float,
    m_Yb: float,
    delta: float,
    omega_R: float,
    U: complex,
    kinetic: bool = True,
    interaction: bool = True,
) -> jsparse.JAXSparse:
    data = []
    rows = []
    cols = []

    if kinetic:
        _v, _r, _c = coo_kinetic_term(hilb, hbar, k_r, m_Yb, delta, omega_R)
        data += _v
        rows += _r
        cols += _c

    if interaction and np.abs(U) > 1e-10:
        _v, _r, _c = coo_interaction_term(hilb, U)
        data += _v
        rows += _r
        cols += _c

    data = jnp.array(data, dtype=jnp.complex128)
    rows = jnp.array(rows, dtype=jnp.int32)
    cols = jnp.array(cols, dtype=jnp.int32)
    indices = jnp.concat([rows, cols], axis=-1)

    hamiltonian_bcoo = jsparse.BCOO(
        (data, indices),
        shape=(hilb.space_dim, hilb.space_dim),
    )
    return jsparse.BCSR.from_bcoo(hamiltonian_bcoo)


def coo_lindblad_two_body_loss(hilb: PBCBox1D, gamma: float):
    coefficent = np.sqrt(gamma / hilb.L)

    data = [[] for _ in range(2 * hilb.n_momentum_points - 1)]
    rows = [[] for _ in range(2 * hilb.n_momentum_points - 1)]
    cols = [[] for _ in range(2 * hilb.n_momentum_points - 1)]

    for state_index, multi_particle_state in enumerate(hilb.multi_particle_states):
        spin_up_states = [
            (idx, q) for idx, (q, sz) in enumerate(multi_particle_state) if sz == 1
        ]
        spin_down_states = [
            (idx, q) for idx, (q, sz) in enumerate(multi_particle_state) if sz == -1
        ]

        for spin_up_state, spin_down_state in itertools.product(
            spin_up_states, spin_down_states
        ):
            idx_up, q1 = spin_up_state
            idx_down, q2 = spin_down_state

            if idx_up < idx_down:
                idx_high = idx_down
                idx_low = idx_up
                n_swaps = idx_down - idx_up
            else:
                idx_high = idx_up
                idx_low = idx_down
                n_swaps = idx_up - idx_down - 1

            new_multi_particle_state = list(multi_particle_state)
            del new_multi_particle_state[idx_high], new_multi_particle_state[idx_low]
            new_idx = hilb.state_to_index_map_multi[tuple(new_multi_particle_state)]

            parity = (-1) ** n_swaps

            op_idx = q1 + q2

            data[op_idx].append(parity * coefficent)
            rows[op_idx].append(new_idx)
            cols[op_idx].append(state_index)

    return data, rows, cols


def lindblad_two_body_loss_scipy_csr(
    hilb: PBCBox1D, gamma: float, reduced_rows: bool = False
):
    data, rows, cols = coo_lindblad_two_body_loss(hilb, gamma)
    n_cols = hilb.space_dim
    if reduced_rows:
        # Exclude subspaces that are not in the range of the loss operator.
        n_rows = hilb.space_dim - math.comb(
            hilb.num_single_particle_states, hilb.n_particles[-1]
        )
    else:
        n_rows = hilb.space_dim
    csrs = [
        scipy.sparse.coo_matrix(
            (_d, (_r, _c)), shape=(n_rows, n_cols), dtype=np.complex128
        ).tocsr()
        for _d, _r, _c in zip(data, rows, cols)
    ]
    return csrs

def lindblad_two_body_loss_jax_bcsr(
    hilb: PBCBox1D, gamma: float, reduced_rows: bool = False
):
    data, rows, cols = coo_lindblad_two_body_loss(hilb, gamma)
    n_cols = hilb.space_dim
    if reduced_rows:
        # Exclude subspaces that are not in the range of the loss operator.
        n_rows = hilb.space_dim - math.comb(
            hilb.num_single_particle_states, hilb.n_particles[-1]
        )
    else:
        n_rows = hilb.space_dim
        
    bcsrs = []
    
    for _d, _r, _c in zip(data, rows, cols):
        # Convert to JAX arrays
        _d = jnp.array(_d, dtype=jnp.complex128)
        _r = jnp.array(_r, dtype=jnp.int32)
        _c = jnp.array(_c, dtype=jnp.int32)
        indices = jnp.concat([_r, _c], axis=-1)
        
        hamiltonian_bcoo = jsparse.BCOO(
            (_d, indices),
            shape=(n_rows, n_cols),
        )
        bcsrs.append(jsparse.BCSR.from_bcoo(hamiltonian_bcoo))
    return bcsrs