# import os
import subprocess
# import itertools
# gamma_T2s = [
    # (0.1, 0.8),
    # (0.25, 0.8),
    # (0.4, 0.8),
    # (0.7, 0.8),
    # (1.0, 0.8),
    # (2.0, 0.8)
# ]

gamma_T2s = [
    (0.25, 1.0),
    (0.25, 2.0),
    (0.25, 4.0),
    (0.25, 10.0),
    (0.25, 20.0),
    (0.25, 50.0)
]

# gamma_T2s = [
#     (0.25, 1e8),
#     (0.40, 1e8),
#     (0.7, 1e8),
#     (1.0, 1e8),
#     (2.0, 1e8)
# ]

# gammas = [
#     0.3,
#     0.4,
#     0.5
# ]
# T2s = [
#     0.8,
#     1.5,
#     1e8
# ]

# gamma_T2s = [
#     (gamma, T2) for gamma, T2 in itertools.product(gammas, T2s)
# ]

print(gamma_T2s)

def make_command(
    *,
    python_file_name: str="src/qjm_lindblad_base.py",
    group_name: str="temp",
    run_name_prefix: str="fig3mixture",
    initial_state: str="soc_ground",
    n_momentum_points: int=13,
    n_particles: int=6,
    n_savesteps: int=61,
    gamma: float=0.4,
    T2: float=0.8,
    sim_time: float=3.0,
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
        # "--jumps_per_step", "4",
        "--gamma", f"{gamma:.4f}",
        "--T2", f"{T2:.4f}",
        "--sim_time", f"{sim_time:.4f}",
        "--seed", f"{seed}"
    ]
    if nh:
        commands.append("--nh")
    return commands

for idx, (gamma, T2) in enumerate(gamma_T2s):
    comm = make_command(gamma=gamma, T2=T2, group_name="fig4c/param_search", run_name_prefix=f"sim_{idx}", nh=True)
    print(f"running simulation for: {gamma=}, {T2=}")
    subprocess.run(comm, check=True)