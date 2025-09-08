import os
import subprocess
import itertools

import numpy as np

# rabi_hamiltonian = np.array([
#     [1.5486 / 2, 1.0585 / 2],
#     [1.0585 / 2, -1.5486 / 2]
# ])

# n_points = 21
# sim_range = (0, 0.8 / 0.1129)
# ts = np.linspace(*sim_range, n_points)

# E, Vt = np.linalg.eigh(rabi_hamiltonian)
# evo_operators = np.einsum("ij,nj,jk->nik", Vt.T.conj(), np.exp(-1j * E[None, :] * ts[:, None]), Vt)

# alphas = evo_operators[:, 0, 0]
# betas = evo_operators[:, 1, 0]

# thetas = 2 * np.arctan2(np.abs(betas), np.abs(alphas))
# phis = np.angle(betas) - np.angle(alphas)
# phis = np.mod(phis, 2 * np.pi)

# theta_phis = np.stack((thetas, phis), axis=1)

n_pars = [
    2,
    3,
    4,
    5,
    6,
    7
]

def make_command(
    *,
    python_file_name: str="src/qjm_lindblad_base.py",
    group_name: str="fig2ab",
    run_name_prefix: str="",
    initial_state: str="superposition",
    theta: float=0,
    phi: float=0,
    n_momentum_points: int=13,
    n_particles: int=6,
    n_savesteps: int=61,
    gamma: float=0.38,
    T2: float=1.25,
    sim_time: float=3.0,
    nh: bool=True,
    seed: int=0
):
    commands = [
        "uv", "run", python_file_name, 
        "--group_name", group_name,
        "--run_name", f"{run_name_prefix}_{{default}}", 
        "--initial_state", initial_state,
        "--n_momentum_points", f"{n_momentum_points}", 
        "--n_particles", f"{n_particles}", 
        "--n_savesteps", f"{n_savesteps}", 
        "--jumps_per_step", "10",
        "--gamma", f"{gamma:.4f}",
        "--T2", f"{T2:.4f}",
        "--theta", f"{theta:.6f}",
        "--phi", f"{phi:.6f}",
        "--sim_time", f"{sim_time:.4f}",
        "--seed", f"{seed}"
    ]
    if nh:
        commands.append("--nh")
    return commands

prob_up = 1 - 0.31844
theta = 2 * np.arccos(np.sqrt(prob_up))  # p(up) = 2/3, p(down) = 1/3
phi = 0

for idx, n in enumerate(n_pars):
    print(f"running simulation {idx}: {n=}")
    comm = make_command(run_name_prefix=f"{idx}", theta=theta, phi=phi, n_particles=n)
    subprocess.run(comm, check=True)