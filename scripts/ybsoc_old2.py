import os
import itertools
import functools
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

import scipy
import jax.experimental.sparse as jsparse

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
    print(f"{format_size((len(multi_particle_states) ** 2) * 16)} for dense representation of an operator (complex128)")

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


def get_momentums(k0: float, n: int):
    n_min = -((n + 1) // 2 - 1)
    n_max = n_min + n - 1
    momentums = k0 * np.linspace(n_min, n_max, n)
    return list(momentums)


class YbSOCSystem:
    def __init__(
        self,
        L: float,
        n_momentum_points: int,
        n_particle: int,
        hbar: float,
        k_r: float,
        m_Yb: float,
        delta: float,
        omega_R: float,
        U: float
    ):
        self.L = L
        self.n_momentum_points = n_momentum_points
        self.n_particle = n_particle
        self.hbar = hbar
        self.k_r = k_r
        self.m_Yb = m_Yb
        self.delta = delta
        self.omega_R = omega_R
        self.U = U
        
        self._orbital_indices = [i for i in range(n_momentum_points)]
        self._single_particle_states = list(itertools.product(self._orbital_indices, [1, -1]))
        self._multi_particle_states = list(itertools.combinations(self._single_particle_states, n_particle))

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
        return self._momentums
    
    @property
    def n_min(self):
        return -((self.n_momentum_points + 1) // 2 - 1)
    
    @property
    def n_max(self):
        return self.n_min + self.n_momentum_points - 1
    
    @property
    def dense_representation_size(self):
        return (len(self.multi_particle_states) ** 2) * 16
    
        
    def print_info(self):
        print_multi_particle_states_info(self.multi_particle_states)

    def kinetic(self, q_x):
        H = np.array([
            [(self.hbar ** 2) * ((q_x - self.k_r) ** 2) / (2 * self.m_Yb) + self.delta / 2, self.omega_R / 2],
            [self.omega_R / 2, (self.hbar ** 2) * ((q_x + self.k_r) ** 2) / (2 * self.m_Yb) - self.delta / 2]
        ])
        return H

    def trace_e(self, k):
        return ((self.hbar ** 2) / (2 * self.m_Yb)) * ((k - self.k_r) ** 2 + (k + self.k_r) ** 2)

    def delta_e(self, k):
        return ((self.hbar ** 2) / (2 * self.m_Yb)) * ((k - self.k_r) ** 2 - (k + self.k_r) ** 2) + self.delta

    def e1(self, k):
        return (self.trace_e(k) + np.sqrt(self.delta_e(k) ** 2 + self.omega_R ** 2)) / 2

    def e2(self, k):
        return (self.trace_e(k) - np.sqrt(self.delta_e(k) ** 2 + self.omega_R ** 2)) / 2

    def theta(self, k):
        return (1 / 2) * np.arctan(self.omega_R / self.delta_e(k))

    def alpha(self, k):
        return np.cos(self.theta(k))

    def beta(self, k):
        return np.sin(self.theta(k))
    
    def coo_kinetic_term(self):
        data = []
        rows = []
        cols = []
        for state_index, multi_particle_state in enumerate(self.multi_particle_states):
            i = 0
            N = self.n_particle
            while i < N:
                momentum_idx, spin = multi_particle_state[i]
                q_x = self.momentums[momentum_idx]
                if (i + 1 < N) and (momentum_idx == multi_particle_state[i + 1][0]):
                    # if state contains c_up(k) and c_down(k) both
                    val = ((self.hbar ** 2) / (2 * self.m_Yb)) * ((q_x - self.k_r) ** 2 + (q_x + self.k_r) ** 2)
                    data.append(val)
                    rows.append(state_index)
                    cols.append(state_index)
                    i += 2
                else:
                    # state contains either c_up(k) or c_down(k), but not both.
                    spin_changed = list(multi_particle_state)   # copy state
                    spin_changed[i] = (momentum_idx, -spin)     # reflect spin
                    spin_changed_tuple = tuple(spin_changed)
                    spin_changed_index = self.state_to_index_map_multi[spin_changed_tuple]
                    
                    # diagonal term
                    val = ((self.hbar ** 2) / (2 * self.m_Yb)) * ((q_x - spin * self.k_r) ** 2) + spin * self.delta / 2
                    data.append(val)
                    rows.append(state_index)
                    cols.append(state_index)
                    
                    # off-diagonal term
                    val = self.omega_R / 2
                    data.append(val)
                    rows.append(spin_changed_index)
                    cols.append(state_index)
                    i += 1
                    
        return data, rows, cols


    def coo_interaction_term(self):
        data = []
        rows = []
        cols = []
        for state_index, multi_particle_state in enumerate(self.multi_particle_states):
            spin_up_states = [(idx, single[0]) for idx, single in enumerate(multi_particle_state) if single[1] == 1]
            spin_down_states = [(idx, single[0]) for idx, single in enumerate(multi_particle_state) if single[1] == -1]
            
            for spin_up_state, spin_down_state in itertools.product(spin_up_states, spin_down_states):
                idx1, momentum_1_idx_i = spin_up_state
                idx2, momentum_2_idx_i = spin_down_state

                for q_idx in self.orbital_indices:
                    k1 = momentum_1_idx_i + self.n_min
                    k2 = momentum_2_idx_i + self.n_min
                    q = q_idx + self.n_min
                    
                    if k1 + q < self.n_min or k1 + q > self.n_max or k2 - q < self.n_min or k2 - q > self.n_max:
                        continue
                    momentum_1_idx_f = k1 + q - self.n_min
                    momentum_2_idx_f = k2 - q - self.n_min
                        
                    new_multi_particle_state = list(multi_particle_state)
                    new_multi_particle_state[idx1] = (momentum_1_idx_f, 1)
                    new_multi_particle_state[idx2] = (momentum_2_idx_f, -1)
                    new_multi_particle_state, parity = sorted_multi_particle_state(new_multi_particle_state)
                    if parity == 0:
                        continue
                    new_idx = self.state_to_index_map_multi[new_multi_particle_state]
                    
                    data.append(parity * (-self.U) / self.L)
                    rows.append(new_idx)
                    cols.append(state_index)
                    
        return data, rows, cols
        

    def dense_hamiltonian(
        self,
        kinetic: bool = True,
        interaction: bool = True
    ):
        hamiltonian = np.zeros((self.space_dim, self.space_dim), np.complex128)
        
        if kinetic:
            data, rows, cols = self.coo_kinetic_term()
            for val, row, col in zip(data, rows, cols):
                hamiltonian[row, col] += val

        if interaction:
            data, rows, cols = self.coo_interaction_term()
            for val, row, col in zip(data, rows, cols):
                hamiltonian[row, col] += val
                
        return hamiltonian
    

    def sparse_hamiltonian_scipy_csr(
        self,
        kinetic: bool = True,
        interaction: bool = True
    ):
        data = []
        rows = []
        cols = [] 
        
        if kinetic:
            _v, _r, _c = self.coo_kinetic_term()
            data += _v
            rows += _r
            cols += _c

        if interaction and np.abs(self.U) > 1e-10:
            _v, _r, _c = self.coo_interaction_term()
            data += _v
            rows += _r
            cols += _c
            
        hamiltonian = scipy.sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(self.space_dim, self.space_dim),
            dtype=np.complex128
        )
        return hamiltonian.tocsr()
    
    def sparse_hamiltonian_jax_bcsr(
        self,
        kinetic: bool = True,
        interaction: bool = True
    ) -> jsparse.JAXSparse:
        data = []
        rows = []
        cols = [] 
        
        if kinetic:
            _v, _r, _c = self.coo_kinetic_term()
            data += _v
            rows += _r
            cols += _c

        if interaction and np.abs(self.U) > 1e-10:
            _v, _r, _c = self.coo_interaction_term()
            data += _v
            rows += _r
            cols += _c
        
        data = jnp.array(data, dtype=jnp.complex128)
        rows = jnp.array(rows, dtype=jnp.int32)
        cols = jnp.array(cols, dtype=jnp.int32)
        indices = jnp.concat([rows, cols], axis=-1)

        hamiltonian_bcoo = jsparse.BCOO(
            (data, indices),
            shape=(self.space_dim, self.space_dim),
        )
        return jsparse.BCSR.from_bcoo(hamiltonian_bcoo)
        
    # number ops    
    def momentum_num_op_diags(self):
        num_op_diags = np.zeros((self.num_single_particle_states, self.space_dim)) # length: space, 2: spin

        for state_index_single, single_states in enumerate(self.single_particle_states):
            for state_index_multi, multi_particle_state in enumerate(self.multi_particle_states):
                if single_states in multi_particle_state:
                    num_op_diags[state_index_single, state_index_multi] = 1
        
        return num_op_diags

    def momentum_expected_numbers(self, state_vector):
        num_op_diags = self.momentum_sp_num_op_diags()
        probs = np.abs(state_vector) ** 2
        expected_numbers = np.einsum('ij,j->i',num_op_diags, probs)
        nums_spin_up = expected_numbers[0::2]
        nums_spin_down = expected_numbers[1::2]
        return nums_spin_up, nums_spin_down

    def momentum_expected_numbers_vectorized(self, state_vectors):
        num_op_diags = self.momentum_sp_num_op_diags()
        probs = np.abs(state_vectors) ** 2
        expected_numbers = np.einsum('ij,...j->...i',num_op_diags, probs)
        nums_spin_up = expected_numbers[..., 0::2]
        nums_spin_down = expected_numbers[..., 1::2]
        return nums_spin_up, nums_spin_down
    
    def momentum_sp_num_op_diags(self):
        return self.momentum_num_op_diags()
    
    def momentum_sp_expected_numbers(self, state_vector):
        return self.momentum_expected_numbers(state_vector)
    
    def momentum_sp_expected_numbers_vectorized(self, state_vectors):
        return self.momentum_expected_numbers_vectorized(state_vectors)
    
    def momentum_expected_numbers_mixed(self, density_matrix):
        num_op_diags = self.momentum_sp_num_op_diags()
        probs = np.diag(density_matrix)
        expected_numbers = np.einsum('ij,j->i', num_op_diags, probs)
        nums_spin_up = expected_numbers[0::2]
        nums_spin_down = expected_numbers[1::2]
        return nums_spin_up, nums_spin_down
    
    # TODO: reimplement position-space density
    
    # def position_num_op(self, site_midx, spin):
    #     num_op = np.zeros((self._space_dim, self._space_dim), dtype=np.complex128)
    #     for state_index, multi_particle_state in enumerate(self.multi_particle_states):
    #         for idx1, single_state in enumerate(multi_particle_state):
    #             q_midx, _spin = single_state
    #             if _spin != spin:
    #                 continue
    #             q_idx = self._site_to_index_map[q_midx]
    #             for k_midx in self.sites_multi_indices:
    #                 new_multi_particle_state = list(multi_particle_state)
    #                 new_multi_particle_state[idx1] = (k_midx, spin)
    #                 new_multi_particle_state, parity = sorted_multi_particle_state(new_multi_particle_state)
    #                 if parity == 0:
    #                     continue
    #                 new_idx = self.state_to_index_map_multi[new_multi_particle_state]
    #                 k_idx = self._site_to_index_map[k_midx]
    #                 site_idx = self._site_to_index_map[site_midx]
    #                 num_op[new_idx, state_index] += parity * (1 / self.num_sites) * np.exp(1j * np.dot(self._momentums[k_idx] - self._momentums[q_idx], self._positions[site_idx]))
    #     return num_op

    # def position_expected_numbers_deprecated(self, state_vector):
    #     nums_spin_up = np.zeros(tuple(self.lengths))
    #     nums_spin_down = np.zeros(tuple(self.lengths))
    #     for site_index in self.sites_multi_indices:
    #         nums_spin_up[*site_index] = np.einsum("i,ij,j", state_vector.conj(), self.position_num_op(site_index, 1), state_vector).real
    #         nums_spin_down[*site_index] = np.einsum("i,ij,j", state_vector.conj(), self.position_num_op(site_index, -1), state_vector).real
    #     return nums_spin_up, nums_spin_down
    
    # def _calculate_correlation_matrix(self, state_vector, spin):
    #     """
    #     운동량 공간 상관 행렬 C_kq = <ψ|c†_k c_q|ψ> 를 계산합니다.
    #     이 행렬은 q 상태의 입자를 소멸시키고 k 상태에 생성했을 때의 진폭을 나타냅니다.
    #     """
    #     num_sites = self.num_sites
    #     C = np.zeros((num_sites, num_sites), dtype=np.complex128)
    #     conj_state_vector = state_vector.conj()

    #     # 모든 다입자 상태(기저)를 순회합니다.
    #     for state_index, multi_particle_state in enumerate(self.multi_particle_states):
    #         # 해당 상태의 계수(진폭)가 0에 가까우면 계산을 건너뜁니다.
    #         if np.isclose(state_vector[state_index], 0):
    #             continue

    #         # 상태 내의 각 입자를 순회합니다 (입자 소멸 연산).
    #         for p_idx, single_state in enumerate(multi_particle_state):
    #             q_midx, _spin = single_state
    #             if _spin != spin:
    #                 continue
                
    #             q_idx = self._site_to_index_map[q_midx]
                
    #             # 가능한 모든 최종 운동량 상태를 순회합니다 (입자 생성 연산).
    #             for k_midx in self.sites_multi_indices:
    #                 new_multi_particle_state = list(multi_particle_state)
    #                 new_multi_particle_state[p_idx] = (k_midx, spin)
                    
    #                 # 페르미온 규칙에 따라 상태를 정렬하고 부호(parity)를 얻습니다.
    #                 # 동일한 상태가 이미 존재하면 parity는 0이 되어 파울리 배타 원리가 적용됩니다.
    #                 new_multi_particle_state, parity = sorted_multi_particle_state(tuple(new_multi_particle_state))

    #                 if parity == 0:
    #                     continue

    #                 new_idx = self.state_to_index_map_multi[new_multi_particle_state]
    #                 k_idx = self._site_to_index_map[k_midx]
                    
    #                 # C_kq += <ψ|new_state> * <state|ψ> * parity
    #                 C[k_idx, q_idx] += parity * conj_state_vector[new_idx] * state_vector[state_index]
                    
    #     return C

    # def position_expected_numbers(self, state_vector):
    #     """
    #     상관 행렬을 푸리에 변환하여 모든 위치에서의 입자 점유 수를 효율적으로 계산합니다.
    #     """
    #     # 1. 스핀별로 상관 행렬을 계산합니다.
    #     C_up = self._calculate_correlation_matrix(state_vector, 1)
    #     C_down = self._calculate_correlation_matrix(state_vector, -1)

    #     # 2. 푸리에 변환 행렬을 미리 계산합니다: F_jk = exp(i * r_j . p_k)
    #     # self._positions와 self._momentums가 site_idx 순서로 정렬되어 있다고 가정합니다.
    #     fourier_matrix = np.exp(1j * np.array(self._positions) @ np.array(self.momentums).T)

    #     # 3. einsum을 사용하여 모든 사이트에 대한 기댓값을 한번에 계산합니다.
    #     # 이 식은 diag(F @ C @ F.conj().T) 와 동일하지만 더 효율적입니다.
    #     # Sum_{k,q} F_jk * C_kq * F*_jq
    #     nums_up_flat = (1 / self.num_sites) * np.einsum('jk,kq,jq->j', fourier_matrix, C_up, fourier_matrix.conj(), optimize=True)
    #     nums_down_flat = (1 / self.num_sites) * np.einsum('jk,kq,jq->j', fourier_matrix, C_down, fourier_matrix.conj(), optimize=True)
        
    #     # 4. 계산 결과를 실제 시스템 모양에 맞게 변환합니다.
    #     nums_spin_up = nums_up_flat.real.reshape(self.lengths)
    #     nums_spin_down = nums_down_flat.real.reshape(self.lengths)

    #     return nums_spin_up, nums_spin_down

    
    
    # def _calculate_correlation_matrix_vectorized(self, state_vectors, spin):
    #     """
    #     상태 벡터들의 배치(batch)에 대한 운동량 공간 상관 텐서 C_{...,k,q} 를 계산합니다.
    #     """
    #     batch_shape = state_vectors.shape[:-1]
    #     num_sites = self.num_sites
    #     C = np.zeros((*batch_shape, num_sites, num_sites), dtype=np.complex128)
    #     conj_state_vectors = state_vectors.conj()

    #     # 모든 다입자 상태(기저)를 순회합니다.
    #     for state_index, multi_particle_state in enumerate(self.multi_particle_states):
    #         # 상태 내의 각 입자를 순회합니다 (입자 소멸 연산).
    #         for p_idx, single_state in enumerate(multi_particle_state):
    #             q_midx, _spin = single_state
    #             if _spin != spin:
    #                 continue
                
    #             q_idx = self._site_to_index_map[q_midx]
                
    #             # 가능한 모든 최종 운동량 상태를 순회합니다 (입자 생성 연산).
    #             for k_midx in self.sites_multi_indices:
    #                 new_multi_particle_state = list(multi_particle_state)
    #                 new_multi_particle_state[p_idx] = (k_midx, spin)
                    
    #                 new_multi_particle_state, parity = sorted_multi_particle_state(tuple(new_multi_particle_state))

    #                 if parity == 0:
    #                     continue

    #                 new_idx = self.state_to_index_map_multi[new_multi_particle_state]
    #                 k_idx = self._site_to_index_map[k_midx]
                    
    #                 # C 행렬의 배치 차원에 대해 벡터화된 연산을 수행합니다.
    #                 # C[b, k, q] += <ψ_b|new_state> * <state|ψ_b> * parity
    #                 C[..., k_idx, q_idx] += parity * conj_state_vectors[..., new_idx] * state_vectors[..., state_index]
                    
    #     return C
    
    # def position_expected_numbers_vectorized(self, state_vectors):
    #     C_up = self._calculate_correlation_matrix_vectorized(state_vectors, 1)
    #     C_down = self._calculate_correlation_matrix_vectorized(state_vectors, -1)
    #     # 2. 푸리에 변환 행렬을 미리 계산합니다: F_jk = exp(i * r_j . p_k)
        
    #     # self._momentums 로 변수명을 수정하였습니다.
    #     fourier_matrix = np.exp(1j * np.array(self._positions) @ np.array(self._momentums).T)

    #     # 3. einsum을 사용하여 배치 푸리에 변환을 한번에 수행합니다.
    #     # '...kq' 와 '...j' 표기법이 배치 차원을 처리합니다.
    #     nums_up_flat = (1 / self.num_sites) * np.einsum('jk,...kq,jq->...j', fourier_matrix, C_up, fourier_matrix.conj(), optimize=True)
    #     nums_down_flat = (1 / self.num_sites) * np.einsum('jk,...kq,jq->...j', fourier_matrix, C_down, fourier_matrix.conj(), optimize=True)
        
    #     # 4. 계산 결과를 실제 시스템 모양에 맞게 변환합니다.
    #     final_shape = (*state_vectors.shape[:-1], *self.lengths)
    #     nums_spin_up = nums_up_flat.real.reshape(final_shape)
    #     nums_spin_down = nums_down_flat.real.reshape(final_shape)

    #     return nums_spin_up, nums_spin_down
    
    # def position_expected_numbers_mixed(self, density_matrix):
    #     nums_spin_up = np.zeros((self.n_momentum_points,))
    #     nums_spin_down = np.zeros((self.n_momentum_points,))
    #     for site_index in self.orbital_indices:
    #         nums_spin_up[*site_index] = np.einsum("ij,ji", self.position_num_op(site_index, 1), density_matrix).real
    #         nums_spin_down[*site_index] = np.einsum("ij,ji", self.position_num_op(site_index, -1), density_matrix).real
    #     return nums_spin_up, nums_spin_down
    
    # TODO: these codes can be optimized by using vector-vector product instead of vector-matrix-vector product
    
    
    def num_momentum_diag(self, momentum_multi_index, spin):
        # return diagonal elements of momentum-space number operator
        diag = np.zeros((self.space_dim,), dtype=np.complex128)
        for index, multi_particle_state in enumerate(self.multi_particle_states):
            if (momentum_multi_index, spin) in multi_particle_state:
                diag[index] = 1
        return diag
    
    def from_momentum_nums(self, multi_particle_state):
        if len(multi_particle_state) != self.n_particle:
            raise ValueError(f"number of the particle setted in the system({self.n_particle}) and given({len(multi_particle_state)}) are mismatched.")
        
        new_multi_particle_state, parity = sorted_multi_particle_state(multi_particle_state)

        if parity == 0:
            raise ValueError("Pauli Exclusion principle is violated.")
        
        index = self.state_to_index_map_multi[new_multi_particle_state]
        state_vector = np.zeros((self.space_dim,), dtype=np.complex128)
        state_vector[index] = parity
        return state_vector
        
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
        L: float,
        n_momentum_points: int,
        n_particle: int,
        hbar: float,
        k_r: float,
        m_Yb: float,
        delta: float,
        omega_R: float,
        gamma: float
    ):
        super().__init__(
            L, n_momentum_points, n_particle, hbar, k_r, m_Yb, delta, omega_R, 1j * gamma / 2
        )
        self._gamma = gamma

    @property
    def gamma(self):
        return self._gamma
    
    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma
        self._U = 1j * gamma / 2
    