import os
import time
import gc
import itertools
import functools
import operator
import pickle
from pprint import pprint

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
jax.config.update("jax_enable_x64", True)
from jax.extend.backend import get_backend
import jax.numpy as jnp

import numpy as np
import tqdm
import matplotlib.pyplot as plt

from ybsoc import *

simulation_name = "hubbard"
data_dir = "/home/pco0511/yb-soc-two-body-loss/data"
root_path = os.path.join(data_dir, simulation_name)

metadata_path = os.path.join(root_path, "metadata.pkl")
eigenvector_path = os.path.join(root_path, "eigenvectors")
eigenvalue_path = os.path.join(root_path, "eigenvalues")
mics_path = os.path.join(root_path, "miscellaneous")

metadata = {
    "fixed": {
        "d":1,
        "lengths":[10,],
        "n_particle":4,
        "hbar":1,
        "q":0.03,
        "m_Yb":1,
        "delta":0.08,
        "omega_R":0.2,
        # "gamma":0.1,
    },
    "unfixed": {
        "U": list(np.linspace(-1, 1, 101)),
    },
}

os.makedirs(eigenvector_path, exist_ok=True)
os.makedirs(eigenvalue_path, exist_ok=True)
os.makedirs(mics_path, exist_ok=True)

data_len = functools.reduce(operator.mul, [len(p) for p in metadata["unfixed"].values()], 1)

min_E_i_sus = np.empty((data_len,))
max_E_i_sus = np.empty((data_len,))
min_E_r_sus = np.empty((data_len,))

for idx, unfixed_params in tqdm.tqdm(enumerate(itertools.product(*metadata["unfixed"].values())), total=data_len):
    keys = metadata["unfixed"].keys()
    unfixed_params = dict(zip(keys, unfixed_params))
    system = YbSOCSystem(
        **metadata["fixed"],
        **unfixed_params,
        array_type='jax'
    )
    hamiltonian = system.dense_hamiltonian()
    eigenvalues, eigenvectors = jnp.linalg.eigh(hamiltonian)
    # save_data
    np.save(os.path.join(eigenvalue_path, f"{idx}.npy"), np.array(eigenvalues))
    np.save(os.path.join(eigenvector_path, f"{idx}.npy"), np.array(eigenvectors))

    E_r = jnp.real(eigenvalues)
    E_i = jnp.imag(eigenvalues)

    min_E_r_idx = jnp.argmin(E_r)
    min_E_i_idx = jnp.argmin(E_i)
    max_E_i_idx = jnp.argmax(E_i)

    indices = jnp.array([min_E_r_idx, min_E_i_idx, max_E_i_idx])
    
    reduced_eigenvectors = eigenvectors[:, indices]
    
    suseptibility_matrix = system.pair_susceptibility_matrix()
    suseptibility = jnp.einsum('ji,jk,ki->i', reduced_eigenvectors.conj(), suseptibility_matrix, reduced_eigenvectors)
    
    min_E_r_sus[idx] = suseptibility[0].astype(jnp.float64)
    min_E_i_sus[idx] = suseptibility[1].astype(jnp.float64)
    max_E_i_sus[idx] = suseptibility[2].astype(jnp.float64)
    
    del system, hamiltonian, eigenvalues, eigenvectors, reduced_eigenvectors, suseptibility_matrix, suseptibility, E_r, E_i
    gc.collect()

with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)

np.save(os.path.join(mics_path, "min_E_i_sus.npy"), min_E_i_sus)
np.save(os.path.join(mics_path, "max_E_i_sus.npy"), max_E_i_sus)
np.save(os.path.join(mics_path, "min_E_r_sus.npy"), min_E_r_sus)
