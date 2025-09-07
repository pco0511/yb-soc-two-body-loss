import os
import subprocess
import itertools

gamma_T2s = [
    (0.38, 0.85),
    (0.38, 0.9),
    (0.38, 1.0),
    (0.38, 1.25),
    (0.38, 1.5)
]

print(gamma_T2s)

def make_command(
    *,  
    python_file_name: str="src/qjm_lindblad_base.py",
    group_name: str="temp",
    run_name_prefix: str="fig3mixture",
    initial_state: str="mixture",
    n_momentum_points: int=13,
    n_particles: int=6,
    n_savesteps: int=51,
    jumps_per_step: int=4,
    gamma: float=0.4,
    T2: float=0.8,
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
        "--sim_time", f"{sim_time:.4f}",
        "--seed", f"{seed}"
    ]
    if nh:
        commands.append("--nh")
    return commands

for gamma, T2 in gamma_T2s:
    comm = make_command(
        gamma=gamma, 
        T2=T2, 
        group_name="fig3b/param_search", 
        initial_state="mixture",
        sim_time=2.0,
        nh=True
    )
    print(f"running simulation: {gamma=}, {T2=}")
    subprocess.run(comm, check=True)