import functools
import itertools
import os
from typing import Literal

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np
import scipy
import tqdm
from opalg import lattice

# utils


def format_size(size_in_bytes):
    """
    Convert a size in bytes to a human-readable string with units B, KB, MB, or GB.

    Args:
        size_in_bytes (int): Size in bytes. Must be non-negative.

    Returns:
        str: Formatted size with appropriate unit.

    Raises:
        ValueError: If size_in_bytes is negative.

    Examples:
        >>> format_size(123)
        '123.00 B'
        >>> format_size(1024)
        '1.00 KB'
        >>> format_size(1048576)
        '1.00 MB'
        >>> format_size(1073741824)
        '1.00 GB'
        >>> format_size(2**40)
        '1024.00 GB'
    """
    if size_in_bytes < 0:
        raise ValueError("Size must be non-negative")

    units = ["B", "KB", "MB", "GB"]
    unit_index = 0

    # Keep dividing by 1024 to find the appropriate unit, but stop at GB
    while size_in_bytes >= 1024 and unit_index < len(units) - 1:
        size_in_bytes /= 1024
        unit_index += 1

    # Format the number to two decimal places
    return f"{size_in_bytes:.2f} {units[unit_index]}"


def print_multi_particle_states_info(multi_particle_states):

    num_particles = len(multi_particle_states[0])
    print(f"{len(multi_particle_states)} {num_particles}-particle states")
    print(f"{format_size(len(multi_particle_states) * 16)} per state (complex128)")
    print(
        f"{format_size((len(multi_particle_states) ** 2) * 16)} for dense representation of an operator (complex128)"
    )


def create_index_map(elements):
    return {element: idx for idx, element in enumerate(elements)}


def order(lhs, rhs):
    site_index1, spin1 = lhs
    site_index2, spin2 = rhs

    if site_index1 < site_index2:
        return True
    elif site_index1 == site_index2:
        return spin1 > spin2
    return False


key_func = functools.cmp_to_key(order)


def sorted_multi_particle_state(multi_particle_state):
    """return sorted multi_particle_state as tuple and parity of swap count
    partiy = 1 if sorted list is even permutation of original list.
    parity = -1 if it is odd permutation.
    parity = 0 if it cannot be determined. (there exists a pair of identical element.)
    """
    lst = list(multi_particle_state)
    n_particles = len(lst)
    parity = 1
    for i in range(n_particles):
        swapped = False
        for j in range(0, n_particles - i - 1):
            if lst[j] == lst[j + 1]:
                parity = 0
                continue
            if not order(lst[j], lst[j + 1]):
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                swapped = True
                parity *= -1
        if not swapped:
            break
    return tuple(lst), parity


class YbSOCSystem:
    def __init__(
        self,
        d: float,
        lengths: list[int],
        n_particle: int,
        hbar: float,
        q: list[float] | float,
        m_Yb: float,
        delta: float,
        omega_R: float,
        U: float,
        array_type: Literal["numpy", "jax", "torch"] = "numpy",
    ):
        self._d = d
        self._lengths = lengths
        self._n_particle = n_particle
        self._hbar = hbar
        self._q = q
        self._m_Yb = m_Yb
        self._delta = delta
        self._omega_R = omega_R
        self._U = U

        self._dim = len(self._lengths)
        if isinstance(self._q, float):
            self._q = [self._q] * self._dim
        self._q = jnp.array(self._q)
        self._k0 = 2 * jnp.pi / (d * jnp.array(lengths))

        self._sites_multi_indices = lattice.multi_indices(lengths)
        self._single_particle_states = list(
            itertools.product(self._sites_multi_indices, [1, -1])
        )
        self._multi_particle_states = list(
            itertools.combinations(self._single_particle_states, n_particle)
        )

        self._num_sites = len(self._sites_multi_indices)
        self._num_single_particle_states = len(self._single_particle_states)
        self._num_multi_particle_states = len(self._multi_particle_states)
        self._space_dim = self._num_multi_particle_states

        self._site_to_index_map = create_index_map(self._sites_multi_indices)
        self._state_to_index_map_single = create_index_map(self._single_particle_states)
        self._state_to_index_map_multi = create_index_map(self._multi_particle_states)

        self._positions = lattice.position_arr(lengths, d)
        self._momentums = lattice.momentum_arr(lengths, d)
        self._array_type = array_type

    @property
    def d(self):
        return self._d

    @property
    def lengths(self):
        return self._lengths

    @property
    def n_particle(self):
        return self._n_particle

    @property
    def hbar(self):
        return self._hbar

    @property
    def q(self):
        return self._q

    @property
    def m_Yb(self):
        return self._m_Yb

    @property
    def delta(self):
        return self._delta

    @property
    def omega_R(self):
        return self._omega_R

    @property
    def U(self):
        return self._U

    @property
    def dim(self):
        return self._dim

    @property
    def k0(self):
        return self._k0

    @property
    def sites_multi_indices(self):
        return self._sites_multi_indices

    @property
    def single_particle_states(self):
        return self._single_particle_states

    @property
    def multi_particle_states(self):
        return self._multi_particle_states

    @property
    def num_sites(self):
        return self._num_sites

    @property
    def num_single_particle_states(self):
        return self._num_single_particle_states

    @property
    def num_multi_particle_states(self):
        return self._num_multi_particle_states

    @property
    def space_dim(self):
        return self._space_dim

    @property
    def site_to_index_map(self):
        return self._site_to_index_map

    @property
    def state_to_index_map_single(self):
        return self._state_to_index_map_single

    @property
    def state_to_index_map_multi(self):
        return self._state_to_index_map_multi

    @property
    def positions(self):
        return self._positions

    @property
    def momentums(self):
        return self._momentums

    @property
    def dense_representation_size(self):
        return (len(self.multi_particle_states) ** 2) * 16

    def print_info(self):
        print_multi_particle_states_info(self.multi_particle_states)

    @functools.partial(jax.jit, static_argnums=(0,))
    def kinetic(self, k):
        t = jnp.array(
            [
                [
                    (self.hbar**2) * jnp.sum((k - self.q) ** 2) / (2 * self.m_Yb)
                    + self.delta / 2,
                    self.omega_R / 2,
                ],
                [
                    self.omega_R / 2,
                    (self.hbar**2) * jnp.sum((k + self.q) ** 2) / (2 * self.m_Yb)
                    - self.delta / 2,
                ],
            ]
        )
        return jax.device_put(t, device=jax.devices("cpu")[0])

    @functools.partial(jax.jit, static_argnums=(0,))
    def trace_e(self, k):
        return ((self.hbar**2) / (2 * self.m_Yb)) * (
            (k - self.q) ** 2 + (k + self.q) ** 2
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def delta_e(self, k):
        return ((self.hbar**2) / (2 * self.m_Yb)) * (
            (k - self.q) ** 2 - (k + self.q) ** 2
        ) + self.delta

    @functools.partial(jax.jit, static_argnums=(0,))
    def e1(self, k):
        return (self.trace_e(k) + jnp.sqrt(self.delta_e(k) ** 2 + self.omega_R**2)) / 2

    @functools.partial(jax.jit, static_argnums=(0,))
    def e2(self, k):
        return (self.trace_e(k) - jnp.sqrt(self.delta_e(k) ** 2 + self.omega_R**2)) / 2

    @functools.partial(jax.jit, static_argnums=(0,))
    def theta(self, k):
        return (1 / 2) * jnp.arctan(self.omega_R / self.delta_e(k))

    @functools.partial(jax.jit, static_argnums=(0,))
    def alpha(self, k):
        return jnp.cos(self.theta(k))

    @functools.partial(jax.jit, static_argnums=(0,))
    def beta(self, k):
        return jnp.sin(self.theta(k))

    def prepare_return_arr(self, arr: np.typing.NDArray):
        match self._array_type:
            case "numpy":
                return np.array(arr)
            case "jax":
                return jnp.array(arr)
            case "torch":
                return torch.tensor(arr)

    def dense_hamiltonian(
        self,
        display_progress: bool = False,
        umklapp: bool = False,
        kinetic: bool = True,
        interaction: bool = True,
    ):
        hamiltonian = np.zeros((self.space_dim, self.space_dim), np.complex128)

        if kinetic:
            for state_index, multi_particle_state in (
                enumerate(tqdm.tqdm(self.multi_particle_states))
                if display_progress
                else enumerate(self.multi_particle_states)
            ):
                iterator = iter(range(self.n_particle))

                for i in iterator:
                    momentum_idx, spin = multi_particle_state[i]
                    momentum = self.momentums[self.site_to_index_map[momentum_idx]]
                    t = np.array(self.kinetic(momentum))
                    if (i + 1 < self.n_particle) and (
                        momentum_idx == multi_particle_state[i + 1][0]
                    ):
                        # state contain c_up(k), c_down(k) both
                        hamiltonian[state_index, state_index] += t.trace()
                        next(iterator, None)
                    else:
                        # state contain c_up(k) xor c_down(k)
                        spin_changed = list(multi_particle_state)  # copy state
                        spin_changed[i] = (momentum_idx, -spin)  # reflect spin
                        spin_changed_tuple = tuple(spin_changed)
                        spin_changed_index = self.state_to_index_map_multi[
                            spin_changed_tuple
                        ]

                        spin_idx = (1 - spin) // 2  # \sigma = 1 -> 0, \sigma -1 -> 1

                        hamiltonian[state_index, state_index] += t[spin_idx, spin_idx]
                        hamiltonian[spin_changed_index, state_index] += t[
                            1 - spin_idx, spin_idx
                        ]

        # interaction term
        if interaction:
            for state_index, multi_particle_state in (
                enumerate(tqdm.tqdm(self.multi_particle_states))
                if display_progress
                else enumerate(self.multi_particle_states)
            ):
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

                    for q_idx in self.sites_multi_indices:
                        if umklapp:
                            momentum_1_idx_f = lattice.add_multi_indices(
                                momentum_1_idx_i, q_idx, self.lengths
                            )
                            momentum_2_idx_f = lattice.subtract_multi_indices(
                                momentum_2_idx_i, q_idx, self.lengths
                            )
                        else:
                            _momentum_1_idx_i = lattice.fold_bz(
                                momentum_1_idx_i, self.lengths
                            )
                            _momentum_2_idx_i = lattice.fold_bz(
                                momentum_2_idx_i, self.lengths
                            )
                            _q_idx = lattice.fold_bz(q_idx, self.lengths)

                            momentum_1_idx_f = lattice.add_multi_indices(
                                _momentum_1_idx_i, _q_idx
                            )
                            momentum_2_idx_f = lattice.subtract_multi_indices(
                                _momentum_2_idx_i, _q_idx
                            )
                            if not (
                                lattice.check_constrain(momentum_1_idx_f, self.lengths)
                                and lattice.check_constrain(
                                    momentum_2_idx_f, self.lengths
                                )
                            ):
                                continue
                            momentum_1_idx_f = lattice.fold(
                                momentum_1_idx_f, self.lengths
                            )
                            momentum_2_idx_f = lattice.fold(
                                momentum_2_idx_f, self.lengths
                            )

                        new_multi_particle_state = list(multi_particle_state)
                        new_multi_particle_state[idx1] = (momentum_1_idx_f, 1)
                        new_multi_particle_state[idx2] = (momentum_2_idx_f, -1)
                        new_multi_particle_state, parity = sorted_multi_particle_state(
                            new_multi_particle_state
                        )
                        if parity == 0:
                            continue
                        new_idx = self.state_to_index_map_multi[
                            new_multi_particle_state
                        ]
                        hamiltonian[new_idx, state_index] += (
                            parity * (-self.U) / (self.d * self.num_sites)
                        )

        return self.prepare_return_arr(hamiltonian)

    def sparse_hamiltonian_scipy_csr(
        self,
        # format: Literal["bsr", "coo", "csc", "csr", "dia", "dok", "lil"] = "csr",
        umklapp: bool = False,
        kinetic: bool = True,
        interaction: bool = True,
    ):
        data = []
        rows = []
        cols = []

        if kinetic:
            for state_index, multi_particle_state in enumerate(
                tqdm.tqdm(self.multi_particle_states)
            ):
                iterator = iter(range(self.n_particle))

                for i in iterator:
                    momentum_idx, spin = multi_particle_state[i]
                    momentum = self.momentums[self.site_to_index_map[momentum_idx]]
                    t = np.array(self.kinetic(momentum))

                    if (i + 1 < self.n_particle) and (
                        momentum_idx == multi_particle_state[i + 1][0]
                    ):
                        rows.append(state_index)
                        cols.append(state_index)
                        data.append(t.trace())
                        next(iterator, None)
                    else:
                        spin_idx = (1 - spin) // 2

                        # diagonal element
                        rows.append(state_index)
                        cols.append(state_index)
                        data.append(t[spin_idx, spin_idx])

                        # off-diagonal element
                        spin_changed = list(multi_particle_state)
                        spin_changed[i] = (momentum_idx, -spin)
                        spin_changed_tuple = tuple(spin_changed)
                        spin_changed_index = self.state_to_index_map_multi[
                            spin_changed_tuple
                        ]

                        rows.append(spin_changed_index)
                        cols.append(state_index)
                        data.append(t[1 - spin_idx, spin_idx])

        if np.abs(self._U) < 1e-10 or not interaction:
            # if U is negligible, return the kinetic part only
            hamiltonian = scipy.sparse.coo_matrix(
                (data, (rows, cols)),
                shape=(self.space_dim, self.space_dim),
                dtype=np.complex128,
            )
            return hamiltonian.tocsr()

        for state_index, multi_particle_state in enumerate(
            tqdm.tqdm(self.multi_particle_states)
        ):
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

                for q_idx in self.sites_multi_indices:
                    if umklapp:
                        momentum_1_idx_f = lattice.add_multi_indices(
                            momentum_1_idx_i, q_idx, self.lengths
                        )
                        momentum_2_idx_f = lattice.subtract_multi_indices(
                            momentum_2_idx_i, q_idx, self.lengths
                        )
                    else:
                        _momentum_1_idx_i = lattice.fold_bz(
                            momentum_1_idx_i, self.lengths
                        )
                        _momentum_2_idx_i = lattice.fold_bz(
                            momentum_2_idx_i, self.lengths
                        )
                        _q_idx = lattice.fold_bz(q_idx, self.lengths)

                        momentum_1_idx_f = lattice.add_multi_indices(
                            _momentum_1_idx_i, _q_idx
                        )
                        momentum_2_idx_f = lattice.subtract_multi_indices(
                            _momentum_2_idx_i, _q_idx
                        )
                        if not (
                            lattice.check_constrain(momentum_1_idx_f, self.lengths)
                            and lattice.check_constrain(momentum_2_idx_f, self.lengths)
                        ):  # ignore
                            continue
                        momentum_1_idx_f = lattice.fold(momentum_1_idx_f, self.lengths)
                        momentum_2_idx_f = lattice.fold(momentum_2_idx_f, self.lengths)

                    new_multi_particle_state = list(multi_particle_state)
                    new_multi_particle_state[idx1] = (momentum_1_idx_f, 1)
                    new_multi_particle_state[idx2] = (momentum_2_idx_f, -1)
                    new_multi_particle_state, parity = sorted_multi_particle_state(
                        new_multi_particle_state
                    )
                    if parity == 0:
                        continue
                    new_idx = self.state_to_index_map_multi[new_multi_particle_state]

                    rows.append(new_idx)
                    cols.append(state_index)
                    data.append(parity * (-self.U) / (self.d * self.num_sites))

        hamiltonian = scipy.sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(self.space_dim, self.space_dim),
            dtype=np.complex128,
        )
        return hamiltonian.tocsr()

    def sparse_hamiltonian_jax_bcsr(
        self,
        # format: Literal["bcoo", "bcsr"] = "bcsr",
        umklapp: bool = False,
        kinetic: bool = True,
        interaction: bool = True,
    ) -> jsparse.JAXSparse:
        data = []
        indices = []

        if kinetic:
            for state_index, multi_particle_state in enumerate(
                self.multi_particle_states
            ):
                iterator = iter(range(self.n_particle))

                for i in iterator:
                    momentum_idx, spin = multi_particle_state[i]
                    momentum = self.momentums[self.site_to_index_map[momentum_idx]]
                    t = np.array(self.kinetic(momentum))

                    if (i + 1 < self.n_particle) and (
                        momentum_idx == multi_particle_state[i + 1][0]
                    ):
                        indices.append([state_index, state_index])
                        data.append(t.trace())
                        next(iterator, None)
                    else:
                        spin_idx = (1 - spin) // 2

                        # diagonal element
                        indices.append([state_index, state_index])
                        data.append(t[spin_idx, spin_idx])

                        # off-diagonal element
                        spin_changed = list(multi_particle_state)
                        spin_changed[i] = (momentum_idx, -spin)
                        spin_changed_tuple = tuple(spin_changed)
                        spin_changed_index = self.state_to_index_map_multi[
                            spin_changed_tuple
                        ]

                        indices.append([spin_changed_index, state_index])
                        data.append(t[1 - spin_idx, spin_idx])

        if np.abs(self._U) < 1e-10 or not interaction:
            # if U is negligible, return the kinetic part only
            data = jnp.array(data, dtype=jnp.complex128)
            indices = jnp.array(indices, dtype=jnp.int32)

            hamiltonian_bcoo = jsparse.BCOO(
                (data, indices),
                shape=(self.space_dim, self.space_dim),
            )
            return jsparse.BCSR.from_bcoo(hamiltonian_bcoo)

        for state_index, multi_particle_state in enumerate(self.multi_particle_states):
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

                for q_idx in self.sites_multi_indices:
                    if umklapp:
                        momentum_1_idx_f = lattice.add_multi_indices(
                            momentum_1_idx_i, q_idx, self.lengths
                        )
                        momentum_2_idx_f = lattice.subtract_multi_indices(
                            momentum_2_idx_i, q_idx, self.lengths
                        )
                    else:
                        _momentum_1_idx_i = lattice.fold_bz(
                            momentum_1_idx_i, self.lengths
                        )
                        _momentum_2_idx_i = lattice.fold_bz(
                            momentum_2_idx_i, self.lengths
                        )
                        _q_idx = lattice.fold_bz(q_idx, self.lengths)

                        momentum_1_idx_f = lattice.add_multi_indices(
                            _momentum_1_idx_i, _q_idx
                        )
                        momentum_2_idx_f = lattice.subtract_multi_indices(
                            _momentum_2_idx_i, _q_idx
                        )
                        if not (
                            lattice.check_constrain(momentum_1_idx_f, self.lengths)
                            and lattice.check_constrain(momentum_2_idx_f, self.lengths)
                        ):  # ignore
                            continue
                        momentum_1_idx_f = lattice.fold(momentum_1_idx_f, self.lengths)
                        momentum_2_idx_f = lattice.fold(momentum_2_idx_f, self.lengths)

                    new_multi_particle_state = list(multi_particle_state)
                    new_multi_particle_state[idx1] = (momentum_1_idx_f, 1)
                    new_multi_particle_state[idx2] = (momentum_2_idx_f, -1)
                    new_multi_particle_state, parity = sorted_multi_particle_state(
                        new_multi_particle_state
                    )
                    if parity == 0:
                        continue
                    new_idx = self.state_to_index_map_multi[new_multi_particle_state]

                    indices.append([new_idx, state_index])
                    data.append(parity * (-self.U) / (self.d * self.num_sites))

        data = jnp.array(data, dtype=jnp.complex128)
        indices = jnp.array(indices, dtype=jnp.int32)

        hamiltonian_bcoo = jsparse.BCOO(
            (data, indices),
            shape=(self.space_dim, self.space_dim),
        )
        return jsparse.BCSR.from_bcoo(hamiltonian_bcoo)

    # number ops
    def momentum_num_op_diags(self):
        num_op_diags = np.zeros(
            (self.num_single_particle_states, self.space_dim)
        )  # length: space, 2: spin

        for state_index_single, single_states in enumerate(self.single_particle_states):
            for state_index_multi, multi_particle_state in enumerate(
                self.multi_particle_states
            ):
                if single_states in multi_particle_state:
                    num_op_diags[state_index_single, state_index_multi] = 1

        return self.prepare_return_arr(num_op_diags)

    def momentum_expected_numbers(self, state_vector):
        num_op_diags = self.momentum_sp_num_op_diags()
        probs = np.abs(state_vector) ** 2
        expected_numbers = np.einsum("ij,j->i", num_op_diags, probs)
        nums_spin_up = expected_numbers[0::2]
        nums_spin_down = expected_numbers[1::2]
        return self.prepare_return_arr(nums_spin_up), self.prepare_return_arr(
            nums_spin_down
        )

    def momentum_expected_numbers_vectorized(self, state_vectors):
        num_op_diags = self.momentum_sp_num_op_diags()
        probs = np.abs(state_vectors) ** 2
        expected_numbers = np.einsum("ij,...j->...i", num_op_diags, probs)
        nums_spin_up = expected_numbers[..., 0::2]
        nums_spin_down = expected_numbers[..., 1::2]
        return self.prepare_return_arr(nums_spin_up), self.prepare_return_arr(
            nums_spin_down
        )

    def momentum_sp_num_op_diags(self):
        return self.momentum_num_op_diags()

    def momentum_sp_expected_numbers(self, state_vector):
        return self.momentum_expected_numbers(state_vector)

    def momentum_sp_expected_numbers_vectorized(self, state_vectors):
        return self.momentum_expected_numbers_vectorized(state_vectors)

    def momentum_expected_numbers_mixed(self, density_matrix):
        num_op_diags = self.momentum_sp_num_op_diags()
        probs = np.diag(density_matrix)
        expected_numbers = np.einsum("ij,j->i", num_op_diags, probs)
        nums_spin_up = expected_numbers[0::2]
        nums_spin_down = expected_numbers[1::2]
        return self.prepare_return_arr(nums_spin_up), self.prepare_return_arr(
            nums_spin_down
        )

    def position_num_op(self, site_midx, spin):
        num_op = np.zeros((self._space_dim, self._space_dim), dtype=np.complex128)
        for state_index, multi_particle_state in enumerate(self.multi_particle_states):
            for idx1, single_state in enumerate(multi_particle_state):
                q_midx, _spin = single_state
                if _spin != spin:
                    continue
                q_idx = self._site_to_index_map[q_midx]
                for k_midx in self.sites_multi_indices:
                    new_multi_particle_state = list(multi_particle_state)
                    new_multi_particle_state[idx1] = (k_midx, spin)
                    new_multi_particle_state, parity = sorted_multi_particle_state(
                        new_multi_particle_state
                    )
                    if parity == 0:
                        continue
                    new_idx = self.state_to_index_map_multi[new_multi_particle_state]
                    k_idx = self._site_to_index_map[k_midx]
                    site_idx = self._site_to_index_map[site_midx]
                    num_op[new_idx, state_index] += (
                        parity
                        * (1 / self.num_sites)
                        * np.exp(
                            1j
                            * np.dot(
                                self._momentums[k_idx] - self._momentums[q_idx],
                                self._positions[site_idx],
                            )
                        )
                    )
        return num_op

    def position_expected_numbers_deprecated(self, state_vector):
        nums_spin_up = np.zeros(tuple(self.lengths))
        nums_spin_down = np.zeros(tuple(self.lengths))
        for site_index in self.sites_multi_indices:
            nums_spin_up[*site_index] = np.einsum(
                "i,ij,j",
                state_vector.conj(),
                self.position_num_op(site_index, 1),
                state_vector,
            ).real
            nums_spin_down[*site_index] = np.einsum(
                "i,ij,j",
                state_vector.conj(),
                self.position_num_op(site_index, -1),
                state_vector,
            ).real
        return nums_spin_up, nums_spin_down

    def _calculate_correlation_matrix(self, state_vector, spin):
        """
        운동량 공간 상관 행렬 C_kq = <ψ|c†_k c_q|ψ> 를 계산합니다.
        이 행렬은 q 상태의 입자를 소멸시키고 k 상태에 생성했을 때의 진폭을 나타냅니다.
        """
        num_sites = self.num_sites
        C = np.zeros((num_sites, num_sites), dtype=np.complex128)
        conj_state_vector = state_vector.conj()

        # 모든 다입자 상태(기저)를 순회합니다.
        for state_index, multi_particle_state in enumerate(self.multi_particle_states):
            # 해당 상태의 계수(진폭)가 0에 가까우면 계산을 건너뜁니다.
            if np.isclose(state_vector[state_index], 0):
                continue

            # 상태 내의 각 입자를 순회합니다 (입자 소멸 연산).
            for p_idx, single_state in enumerate(multi_particle_state):
                q_midx, _spin = single_state
                if _spin != spin:
                    continue

                q_idx = self._site_to_index_map[q_midx]

                # 가능한 모든 최종 운동량 상태를 순회합니다 (입자 생성 연산).
                for k_midx in self.sites_multi_indices:
                    new_multi_particle_state = list(multi_particle_state)
                    new_multi_particle_state[p_idx] = (k_midx, spin)

                    # 페르미온 규칙에 따라 상태를 정렬하고 부호(parity)를 얻습니다.
                    # 동일한 상태가 이미 존재하면 parity는 0이 되어 파울리 배타 원리가 적용됩니다.
                    new_multi_particle_state, parity = sorted_multi_particle_state(
                        tuple(new_multi_particle_state)
                    )

                    if parity == 0:
                        continue

                    new_idx = self.state_to_index_map_multi[new_multi_particle_state]
                    k_idx = self._site_to_index_map[k_midx]

                    # C_kq += <ψ|new_state> * <state|ψ> * parity
                    C[k_idx, q_idx] += (
                        parity * conj_state_vector[new_idx] * state_vector[state_index]
                    )

        return C

    def position_expected_numbers(self, state_vector):
        """
        상관 행렬을 푸리에 변환하여 모든 위치에서의 입자 점유 수를 효율적으로 계산합니다.
        """
        # 1. 스핀별로 상관 행렬을 계산합니다.
        C_up = self._calculate_correlation_matrix(state_vector, 1)
        C_down = self._calculate_correlation_matrix(state_vector, -1)

        # 2. 푸리에 변환 행렬을 미리 계산합니다: F_jk = exp(i * r_j . p_k)
        # self._positions와 self._momentums가 site_idx 순서로 정렬되어 있다고 가정합니다.
        fourier_matrix = np.exp(
            1j * np.array(self._positions) @ np.array(self.momentums).T
        )

        # 3. einsum을 사용하여 모든 사이트에 대한 기댓값을 한번에 계산합니다.
        # 이 식은 diag(F @ C @ F.conj().T) 와 동일하지만 더 효율적입니다.
        # Sum_{k,q} F_jk * C_kq * F*_jq
        nums_up_flat = (1 / self.num_sites) * np.einsum(
            "jk,kq,jq->j", fourier_matrix, C_up, fourier_matrix.conj(), optimize=True
        )
        nums_down_flat = (1 / self.num_sites) * np.einsum(
            "jk,kq,jq->j", fourier_matrix, C_down, fourier_matrix.conj(), optimize=True
        )

        # 4. 계산 결과를 실제 시스템 모양에 맞게 변환합니다.
        nums_spin_up = nums_up_flat.real.reshape(self.lengths)
        nums_spin_down = nums_down_flat.real.reshape(self.lengths)

        return nums_spin_up, nums_spin_down

    def _calculate_correlation_matrix_vectorized(self, state_vectors, spin):
        """
        상태 벡터들의 배치(batch)에 대한 운동량 공간 상관 텐서 C_{...,k,q} 를 계산합니다.
        """
        batch_shape = state_vectors.shape[:-1]
        num_sites = self.num_sites
        C = np.zeros((*batch_shape, num_sites, num_sites), dtype=np.complex128)
        conj_state_vectors = state_vectors.conj()

        # 모든 다입자 상태(기저)를 순회합니다.
        for state_index, multi_particle_state in enumerate(self.multi_particle_states):
            # 상태 내의 각 입자를 순회합니다 (입자 소멸 연산).
            for p_idx, single_state in enumerate(multi_particle_state):
                q_midx, _spin = single_state
                if _spin != spin:
                    continue

                q_idx = self._site_to_index_map[q_midx]

                # 가능한 모든 최종 운동량 상태를 순회합니다 (입자 생성 연산).
                for k_midx in self.sites_multi_indices:
                    new_multi_particle_state = list(multi_particle_state)
                    new_multi_particle_state[p_idx] = (k_midx, spin)

                    new_multi_particle_state, parity = sorted_multi_particle_state(
                        tuple(new_multi_particle_state)
                    )

                    if parity == 0:
                        continue

                    new_idx = self.state_to_index_map_multi[new_multi_particle_state]
                    k_idx = self._site_to_index_map[k_midx]

                    # C 행렬의 배치 차원에 대해 벡터화된 연산을 수행합니다.
                    # C[b, k, q] += <ψ_b|new_state> * <state|ψ_b> * parity
                    C[..., k_idx, q_idx] += (
                        parity
                        * conj_state_vectors[..., new_idx]
                        * state_vectors[..., state_index]
                    )

        return C

    def position_expected_numbers_vectorized(self, state_vectors):
        C_up = self._calculate_correlation_matrix_vectorized(state_vectors, 1)
        C_down = self._calculate_correlation_matrix_vectorized(state_vectors, -1)
        # 2. 푸리에 변환 행렬을 미리 계산합니다: F_jk = exp(i * r_j . p_k)

        # self._momentums 로 변수명을 수정하였습니다.
        fourier_matrix = np.exp(
            1j * np.array(self._positions) @ np.array(self._momentums).T
        )

        # 3. einsum을 사용하여 배치 푸리에 변환을 한번에 수행합니다.
        # '...kq' 와 '...j' 표기법이 배치 차원을 처리합니다.
        nums_up_flat = (1 / self.num_sites) * np.einsum(
            "jk,...kq,jq->...j",
            fourier_matrix,
            C_up,
            fourier_matrix.conj(),
            optimize=True,
        )
        nums_down_flat = (1 / self.num_sites) * np.einsum(
            "jk,...kq,jq->...j",
            fourier_matrix,
            C_down,
            fourier_matrix.conj(),
            optimize=True,
        )

        # 4. 계산 결과를 실제 시스템 모양에 맞게 변환합니다.
        final_shape = (*state_vectors.shape[:-1], *self.lengths)
        nums_spin_up = nums_up_flat.real.reshape(final_shape)
        nums_spin_down = nums_down_flat.real.reshape(final_shape)

        return nums_spin_up, nums_spin_down
        # nums_spin_up = np.zeros((*state_vectors.shape[:-1], *self.lengths))
        # nums_spin_down = np.zeros((*state_vectors.shape[:-1], *self.lengths))
        # for site_index in self.sites_multi_indices:
        #     nums_spin_up[..., *site_index] = np.einsum("...i,ij,...j->...", state_vectors.conj(), self.position_num_op(site_index, 1), state_vectors).real
        #     nums_spin_down[..., *site_index] = np.einsum("...i,ij,...j->...", state_vectors.conj(), self.position_num_op(site_index, -1), state_vectors).real
        # return nums_spin_up, nums_spin_down

    def position_expected_numbers_mixed(self, density_matrix):
        nums_spin_up = np.zeros(tuple(self.lengths))
        nums_spin_down = np.zeros(tuple(self.lengths))
        for site_index in self.sites_multi_indices:
            nums_spin_up[*site_index] = np.einsum(
                "ij,ji", self.position_num_op(site_index, 1), density_matrix
            ).real
            nums_spin_down[*site_index] = np.einsum(
                "ij,ji", self.position_num_op(site_index, -1), density_matrix
            ).real
        return nums_spin_up, nums_spin_down

    # TODO: these codes can be optimized by using vector-vector product instead of vector-matrix-vector product

    # correlation functions

    def num_momentum_diag(self, momentum_multi_index, spin):
        # return diagonal elements of momentum-space number operator
        zero_vector = np.zeros((self.space_dim,), dtype=np.complex128)
        for index, multi_particle_state in enumerate(self.multi_particle_states):
            if (momentum_multi_index, spin) in multi_particle_state:
                zero_vector[index] = 1
        return self.prepare_return_arr(zero_vector)

    def num_cooper_pairs_diag(self):
        zero_vector = np.zeros((self.space_dim,), dtype=np.complex128)
        for momentum_midx in self.sites_multi_indices:
            momentum_midx_neg = lattice.negate_multi_index(momentum_midx, self.lengths)
            n1 = self.num_momentum_diag(momentum_midx, 1)
            n2 = self.num_momentum_diag(momentum_midx_neg, -1)
            zero_vector += n1 * n2
        return self.prepare_return_arr(zero_vector)

    def cooper_pair_corr_diag(self):
        zero_vector = np.zeros((self.space_dim,), dtype=np.complex128)
        for momentum_midx in self.sites_multi_indices:
            momentum_midx_op = lattice.negate_multi_index(momentum_midx, self.lengths)
            n1 = self.num_momentum_diag(momentum_midx, 1)
            n2 = self.num_momentum_diag(momentum_midx_op, -1)
            zero_vector += 1 - n1 - n2 + n1 * n2
        zero_vector *= ((2 * np.pi) ** self.dim) / (self.num_sites**2)
        return self.prepare_return_arr(zero_vector)

    def pair_susceptibility_matrix(self):
        corr_op = np.zeros((self.space_dim, self.space_dim), dtype=np.complex128)

        for state_index, multi_particle_state in enumerate(self.multi_particle_states):
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

            # c^\dagger_\downarrow(q)c^\dagger_\uparrow(q)c_\uparrow(k)c_downarrow(k)
            for spin_up_state in spin_up_states:
                idx1, momentum_midx = spin_up_state
                oppo_midx = lattice.negate_multi_index(momentum_midx, self.lengths)

                # search (oppo_midx, down)
                idx2 = next((t[0] for t in spin_down_states if t[1] == oppo_midx), None)

                # if unfound, continue
                if idx2 is None:
                    continue

                for q_midx in self.sites_multi_indices:
                    q_midx_neg = lattice.negate_multi_index(q_midx, self.lengths)
                    new_multi_particle_state = list(multi_particle_state)
                    new_multi_particle_state[idx1] = (q_midx, 1)
                    new_multi_particle_state[idx2] = (q_midx_neg, -1)
                    new_multi_particle_state, parity = sorted_multi_particle_state(
                        new_multi_particle_state
                    )
                    if parity == 0:
                        continue
                    new_idx = self.state_to_index_map_multi[new_multi_particle_state]
                    corr_op[new_idx, state_index] += parity

        corr_op *= ((2 * np.pi) ** self.dim) / (self.num_sites**2)

        return self.prepare_return_arr(corr_op)

    # state preparation

    def from_momentum_nums(self, multi_particle_state):
        if len(multi_particle_state) != self.n_particle:
            raise ValueError(
                f"number of the particle setted in the system({self.n_particle}) and given({len(multi_particle_state)}) are mismatched."
            )

        new_multi_particle_state, parity = sorted_multi_particle_state(
            multi_particle_state
        )

        if parity == 0:
            raise ValueError("Pauli Exclusion principle is violated.")

        index = self.state_to_index_map_multi[new_multi_particle_state]
        state_vector = np.zeros((self.space_dim,), dtype=np.complex128)
        state_vector[index] = parity
        return self.prepare_return_arr(state_vector)

    def get_momentum_eigenstate(self, multi_particle_state):
        return self.from_momentum_nums(multi_particle_state)

    def from_position_nums(self, multi_particle_state):
        raise NotImplementedError()

    def from_momentum_wavefunctions(self, wavefunctions):
        raise NotImplementedError()

    def from_position_wavefunctions(self, wavefunctions):
        raise NotImplementedError()


class YbSOC2bodyLoss(YbSOCSystem):
    def __init__(
        self,
        d: float,
        lengths: list[int],
        n_particle: int,
        hbar: float,
        q: list[float] | float,
        m_Yb: float,
        delta: float,
        omega_R: float,
        gamma: float,
        array_type: Literal["numpy", "jax", "torch"] = "numpy",
    ):
        super().__init__(
            d,
            lengths,
            n_particle,
            hbar,
            q,
            m_Yb,
            delta,
            omega_R,
            1j * gamma / 2,
            array_type,
        )
        self._gamma = gamma

    @property
    def gamma(self):
        return self._gamma
