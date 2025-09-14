import os
import subprocess
import itertools
import numpy as np

n_pars = [
    2,
    3,
    4,
    5,
    6
]

print(n_pars)

def make_command(
    *,  
    python_file_name: str="src/qjm_lindblad_base.py",
    group_name: str="temp",
    run_name_prefix: str="fig3mixture",
    initial_state: str="mixture",
    theta: float=0.0,
    phi: float=0.0,
    n_momentum_points: int=13,
    n_particles: int=6,
    n_savesteps: int=51,
    jumps_per_step: int=4,
    gamma: float=0.38,
    T2: float=1.25,
    sim_time: float=2.0,
    nh: bool=False,
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
        "--jumps_per_step", f"{jumps_per_step}",
        "--gamma", f"{gamma:.4f}",
        "--T2", f"{T2:.4f}",
        "--theta", f"{theta:.4f}",
        "--phi", f"{phi:.4f}",
        "--sim_time", f"{sim_time:.4f}",
        "--seed", f"{seed}"
    ]
    if nh:
        commands.append("--nh")
    return commands

for n in n_pars:
    prob_up = 1 - 0.31844
    theta = 2 * np.arccos(np.sqrt(prob_up))  # p(up) = 2/3, p(down) = 1/3
    phi = 0
    comm = make_command(n_particles=n, group_name="fig3b/param_search", initial_state="superposition", theta=theta, phi=phi, nh=True)
    print(f"running simulation: {n=}")
    subprocess.run(comm, check=True)