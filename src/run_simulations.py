import os
import subprocess

gamma_T2s = [
    (0.1, 0.8),
    (0.25, 0.8),
    (0.4, 0.8),
    (0.7, 0.8),
    (1.0, 0.8),
    (2.0, 0.8)
]

def make_command(gamma, T2):
    commands = [
        "uv", "run", "src/fig4-eachtraj.py", 
        "--run_name", "trend_{default}", 
        "--n_momentum_points", "13", 
        "--n_particles", "6", 
        "--n_savesteps", "61", 
        "--gamma", f"{gamma:.2f}",
        "--T2", f"{T2:.2f}",
        "--seed", "0"
    ]
    return commands

for gamma, T2 in gamma_T2s:
    comm = make_command(gamma, T2)
    print(f"running simulation: {gamma=}, {T2=}")
    subprocess.run(comm, check=True)