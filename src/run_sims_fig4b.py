import os
import subprocess
import itertools
import numpy as np

index_to_load = list(range(0, 201, 20))

print(index_to_load)

def make_command(
    *,  
    python_file_name: str="src/qjm_lindblad_fig4b_base.py",
    group_name: str="fig4b",
    run_name_prefix: str="",
    nh: bool=True,
    soc_dir: str="/workspace/yb-soc-two-body-loss/data/soc_evolutions/13_5_09081644",
    initial_state_index: int=0,
    n_savesteps: int=51,
    n_trajectories: int=512,
    batch_size: int=64,
    jumps_per_step: int=4,
    gamma: float=0.38,
    T2: float=1.25,
    sim_time: float=2.0,
    seed: int=0
):
    commands = [
        "uv", "run", python_file_name,
        '--group_name', group_name,
        '--run_name', f"{run_name_prefix}_{{default}}",
        '--soc_dir', soc_dir,
        '--initial_state_index', f"{initial_state_index}",
        '--n_savesteps', f"{n_savesteps}",
        '--n_trajectories', f"{n_trajectories}",
        '--batch_size', f"{batch_size}",
        '--jumps_per_step', f"{jumps_per_step}",
        '--gamma', f"{gamma:.4f}",
        '--T2', f"{T2:.4f}",
        '--sim_time', f"{sim_time:.4f}",
        '--seed', f"{seed}"
    ]
    if nh:
        commands.append("--nh")
    return commands

for i, iidx in enumerate(index_to_load):
    comm = make_command(initial_state_index=iidx, run_name_prefix=f"run_{i}")
    print(f"running simulation: {iidx=}")
    subprocess.run(comm, check=True)