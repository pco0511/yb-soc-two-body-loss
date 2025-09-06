import os
import subprocess
import itertools

gamma_T2s = [
    # (0.1, 0.8),
    # (0.25, 0.8),
    # (0.4, 0.8),
    # (0.7, 0.8),
    # (1.0, 0.8),
    (2.0, 0.8)
]

# gamma_T2s = [
#     # (0.25, 0.2),
#     # (0.25, 0.4),
#     # (0.25, 1.0),
#     # (0.25, 2.0),
#     # (0.25, 4.0),
#     (0.25, 300.0)
# ]

# gamma_T2s = [
#     (0.25, 1e8),
#     (0.40, 1e8),
#     (0.7, 1e8),
#     (1.0, 1e8),
#     (2.0, 1e8)
# ]

def make_command(gamma, T2):
    commands = [
        "uv", "run", "src/fig4-eachtraj.py", 
        "--run_name", "highgammafine_{default}", 
        "--n_momentum_points", "13", 
        "--n_particles", "6", 
        "--n_savesteps", "101", 
        "--jumps_per_step", "4",
        "--gamma", f"{gamma:.2f}",
        "--T2", f"{T2:.2f}",
        "--sim_time", "1.0",
        "--seed", "0"
    ]
    return commands

for gamma, T2 in gamma_T2s:
    comm = make_command(gamma, T2)
    print(f"running simulation: {gamma=}, {T2=}")
    subprocess.run(comm, check=True)