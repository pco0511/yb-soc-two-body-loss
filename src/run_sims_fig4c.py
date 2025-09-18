import os
import subprocess
import itertools

temperatures = [
    0.025,
    0.05,
    0.1,
    0.25,
    0.5
]

print(temperatures)

def make_command(
    *,
    python_file_name: str="src/qjm_lindblad_base.py",
    group_name: str="fig4c0919",
    run_name_prefix: str="temp",
    nh: bool=True,
    initial_state: str="soc_ground",
    theta: float=0.0,
    phi: float=0.0,
    n_momentum_points: int=13,
    n_particles: int=6,
    n_savesteps: int=101,
    n_trajectories: int=1024,
    batch_size: int=64,
    jumps_per_step: int=4,
    gamma: float=0.38,
    T2: float=1.25,
    temperature: None | float=None,
    sim_time: float=4.0,
    seed: int=0
):
    commands = [
        "uv", "run", python_file_name, 
        "--group_name", group_name,
        "--run_name", f"{run_name_prefix}_{{default}}", 
        "--initial_state", initial_state,
        "--theta", f"{theta:.4f}",
        "--phi", f"{phi:.4f}",
        "--n_momentum_points", f"{n_momentum_points}", 
        "--n_particles", f"{n_particles}", 
        "--n_savesteps", f"{n_savesteps}",
        "--n_trajectories", f"{n_trajectories}",
        "--batch_size", f"{batch_size}",
        "--jumps_per_step", f"{jumps_per_step}",
        "--gamma", f"{gamma:.4f}",
        "--T2", f"{T2:.4f}",
        "--sim_time", f"{sim_time:.4f}",
        "--seed", f"{seed}"
    ]
    if nh:
        commands.append("--nh")
    if temperature is not None:
        commands.extend(["--temperature", f"{temperature:.4f}"])
    return commands

for temp in temperatures:
    comm = make_command(
        temperature=temp
    )
    print(f"running simulation: T={temp}")
    subprocess.run(comm, check=True)