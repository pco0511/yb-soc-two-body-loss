import itertools

import numpy as np
from einops import einsum, rearrange

from .utils import (
    create_index_map,
    print_multi_particle_states_info,
    sorted_multi_particle_state,
)


def get_momentums(k0: float, n: int):
    n_min = -((n + 1) // 2 - 1)
    n_max = n_min + n - 1
    momentums = k0 * np.linspace(n_min, n_max, n)
    return list(momentums)


class PBCBox1D:
    def __init__(
        self,
        L: float,
        n_momentum_points: int,
        n_particles: int | list[int],
    ):
        self.L = L
        self.n_momentum_points = n_momentum_points

        if not (
            isinstance(n_particles, int)
            or (
                isinstance(n_particles, list)
                and all(isinstance(x, int) for x in n_particles)
            )
        ):
            raise ValueError("n_particles should be an integer or a list of integers.")
        if isinstance(n_particles, list):
            if not all((isinstance(x, int) and x >= 0) for x in n_particles):
                raise ValueError(
                    "All elements in n_particles list should be non-negative integers."
                )
            self.n_particles = sorted(n_particles)
        elif isinstance(n_particles, int):
            if n_particles < 0:
                raise ValueError("n_particles should be a non-negative.")
            self.n_particles = n_particles
        else:
            raise ValueError("n_particles should be an integer or a list of integers.")

        self._orbital_indices = [i for i in range(n_momentum_points)]
        self._single_particle_states = list(
            itertools.product(self._orbital_indices, [1, -1])
        )
        if isinstance(self.n_particles, int):
            self._multi_particle_states = list(
                itertools.combinations(self._single_particle_states, self.n_particles)
            )
        else:
            self._multi_particle_states = []
            for n in self.n_particles:
                self._multi_particle_states.extend(
                    itertools.combinations(self._single_particle_states, n)
                )

        self._num_single_particle_states = len(self._single_particle_states)
        self._num_multi_particle_states = len(self._multi_particle_states)

        self._site_to_index_map = create_index_map(self._orbital_indices)
        self._state_to_index_map_single = create_index_map(self._single_particle_states)
        self._state_to_index_map_multi = create_index_map(self._multi_particle_states)

        self._momentums = get_momentums(self.k0, self.n_momentum_points)

    @property
    def k0(self):
        return 2 * np.pi / self.L

    @property
    def orbital_indices(self):
        return self._orbital_indices

    @property
    def single_particle_states(self):
        return self._single_particle_states

    @property
    def multi_particle_states(self):
        return self._multi_particle_states

    @property
    def num_single_particle_states(self):
        return self._num_single_particle_states

    @property
    def num_multi_particle_states(self):
        return self._num_multi_particle_states

    @property
    def space_dim(self):
        return self._num_multi_particle_states

    @property
    def state_to_index_map_single(self):
        return self._state_to_index_map_single

    @property
    def state_to_index_map_multi(self):
        return self._state_to_index_map_multi

    @property
    def momentums(self):
        return np.array(self._momentums)

    @property
    def n_min(self):
        return -((self.n_momentum_points + 1) // 2 - 1)

    @property
    def n_max(self):
        return self.n_min + self.n_momentum_points - 1

    @property
    def k_min(self):
        return self.k0 * self.n_min

    @property
    def k_max(self):
        return self.k0 * self.n_max

    @property
    def dense_representation_size(self):
        return (len(self.multi_particle_states) ** 2) * 16

    def print_info(self):
        print_multi_particle_states_info(self.multi_particle_states)

    def check_n_particles(self, n_particles: int):
        if isinstance(self.n_particles, int):
            return n_particles == self.n_particles
        elif isinstance(self.n_particles, list):
            return n_particles in self.n_particles
        return False

    # number ops
    def momentum_num_op_diags(self):
        num_op_diags = np.zeros(
            (self.num_single_particle_states, self.space_dim)
        )  # length: space, 2: spin

        for state_index, multi_particle_state in enumerate(self.multi_particle_states):
            for single_state in multi_particle_state:
                site_index = self.state_to_index_map_single[single_state]
                num_op_diags[site_index, state_index] = 1

        return num_op_diags

    def momentum_expected_numbers(self, state_vectors):
        num_op_diags = self.momentum_num_op_diags()
        probs = np.abs(state_vectors) ** 2
        expected_numbers = np.einsum("ij,...j->...i", num_op_diags, probs)
        return rearrange(expected_numbers, "... (m spin) -> ... spin m", spin=2)

    def momentum_expected_numbers_mixed(self, density_matrix):
        num_op_diags = self.momentum_num_op_diags()
        expected_numbers = np.einsum("ij,...jj->...i", num_op_diags, density_matrix)
        return rearrange(expected_numbers, "... (m spin) -> ... spin m", spin=2)

    def rdm_sum(self, state_vectors):
        # \bra{\psi} c^\dagger_\sigma(k) c_\sigma(q) \ket{\psi}
        batch_shape = state_vectors.shape[:-1]
        sums = np.zeros((*batch_shape, 2, self.n_momentum_points), dtype=np.complex128)

        for state_index, multi_particle_state in enumerate(self.multi_particle_states):

            # diagonal elements
            for q_idx, sz in multi_particle_state:
                spin_idx = (1 - sz) // 2  # 0 for spin up, 1 for spin down
                sums[..., spin_idx, 0] += np.abs(state_vectors[..., state_index]) ** 2

            # off-diagonal elements
            for idx, (q_idx, sz) in enumerate(multi_particle_state):
                spin_idx = (1 - sz) // 2  # 0 for spin up, 1 for spin down

                # loop for (k > q)
                for k_idx in range(q_idx + 1, self.n_momentum_points):
                    if (k_idx, sz) in multi_particle_state:
                        continue

                    temp_state = list(multi_particle_state)
                    temp_state[idx] = (k_idx, sz)
                    temp_state, parity = sorted_multi_particle_state(temp_state)

                    assert parity != 0, "Pauli Exclusion principle is violated."

                    new_state_index = self.state_to_index_map_multi[temp_state]
                    term = (
                        parity
                        * state_vectors[..., new_state_index].conj()
                        * state_vectors[..., state_index]
                    )
                    delta = k_idx - q_idx
                    sums[..., spin_idx, delta] += term

        return sums

    def position_expected_numbers(self, state_vectors):
        sums = self.rdm_sum(state_vectors)

        xs = np.linspace(0, self.L, self.n_momentum_points, endpoint=False) + self.L / (
            2 * self.n_momentum_points
        )
        deltas = self.momentums - self.k_min

        fourier = np.exp(1j * einsum(xs, deltas, "xs, deltas -> xs deltas"))
        densities = 2 * np.real(
            einsum(fourier, sums, "xs deltas, ... spins deltas -> ... spins xs")
        ) - np.expand_dims(np.real(sums[..., 0]), axis=-1)
        densities /= self.L

        return densities

    # TODO: functions for mixed state

    def num_momentum_diag(self, momentum_multi_index, spin):
        # return diagonal elements of momentum-space number operator
        diag = np.zeros((self.space_dim,), dtype=np.complex128)
        for index, multi_particle_state in enumerate(self.multi_particle_states):
            if (momentum_multi_index, spin) in multi_particle_state:
                diag[index] = 1
        return diag

    def from_momentum_occupations(self, multi_particle_state):
        if not self.check_n_particles(len(multi_particle_state)):
            raise ValueError(
                f"number of the particle setted in the system({self.n_particles}) and given({len(multi_particle_state)}) are mismatched."
            )

        new_multi_particle_state, parity = sorted_multi_particle_state(
            multi_particle_state
        )

        if parity == 0:
            raise ValueError("Pauli Exclusion principle is violated.")

        index = self.state_to_index_map_multi[new_multi_particle_state]
        state_vector = np.zeros((self.space_dim,), dtype=np.complex128)
        state_vector[index] = parity
        return state_vector

    def get_momentum_eigenstate(self, multi_particle_state):
        return self.from_momentum_occupations(multi_particle_state)

    def from_momentum_wavefunctions(self, wavefunctions):
        raise NotImplementedError()

    def from_position_wavefunctions(self, wavefunctions):
        raise NotImplementedError()
