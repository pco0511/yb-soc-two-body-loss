import time
import math
import json
import os
import itertools
from pathlib import Path
from functools import partial

import numpy as np
import scipy

import tqdm
import matplotlib.pyplot as plt
import gc

from quant_mech.jax_linalg.expm_utils import (
    expm_steps_est
)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

from jaxtyping import Array, Complex, PRNGKeyArray

from einops import einsum, rearrange, unpack

from quant_mech import hilbert, ybsoc
from quant_mech.plot_utils import (
    plot_numbers,
    generate_k_labels,
    plot_fractions
)
from quant_mech.jax_linalg.expm_utils import apply_expm
from quant_mech.qjm import qjm_step
import argparse

print(f"Using: {jax.devices()}")


parser = argparse.ArgumentParser(description="Time evolution of SOC Hamiltonian")

parser.add_argument('--group_name', type=str, default="soc_evolutions")
parser.add_argument('--run_name', type=str, help='Root directory for data and figures')
parser.add_argument('--n_savesteps', type=int, default=201, help='Number of time steps to save')
parser.add_argument('--n_momentum_points', type=int, required=True, help='Number of momentum points')
parser.add_argument('--n_particles', type=int, required=True, help='Number of particles')
parser.add_argument('--taylor_order', type=int, default=8, help='order of taylor polynomial for expm calculation')
parser.add_argument('--sim_time', type=float, default=0.4, help='simulation time in mili second')
# parser.add_argument('--delta', type=float, default=4.0, help='delta')
# parser.add_argument('--omega_R', type=float, default=3.5, help='omega_R')


parser.add_argument('--seed', type=int, help='Random seed if not given, the timestamp will be used.')



args = parser.parse_args()

# Generate timestamp (MMDDHHMM)
CURRENT_TIME_STAMP = time.strftime("%m%d%H%M")

DATA_ROOT = Path(os.path.dirname(__file__)).parent / "data" / args.group_name
DEFAULT_NAME = f"{args.n_momentum_points}_{args.n_particles}_{CURRENT_TIME_STAMP}"
        
# Use parsed arguments
if args.run_name:
    if "{default}" in args.run_name:
        RUN_NAME = args.run_name.replace("{default}", DEFAULT_NAME)
    else:
        RUN_NAME = args.run_name
else:
    RUN_NAME = DEFAULT_NAME
    
DATA_ROOT = DATA_ROOT / RUN_NAME

data_dir = os.path.join(DATA_ROOT, "data")
figs_dir = os.path.join(DATA_ROOT, "figs")
print(f"saving data to: {DATA_ROOT}")

if args.seed is not None:
    seed = args.seed
else:
    seed = int(time.time())
    
key = jax.random.key(seed)

os.makedirs(data_dir, exist_ok=True)
os.makedirs(figs_dir, exist_ok=True)

# System definition
t_r = 0.1129 # ms
E_r = 9.3428e-31 # J
T_r = 6.7669e-2 # μK

n_momentum_points = args.n_momentum_points
n_particles = args.n_particles

hbar = 1.
m_Yb = 1.

k_r = 1.0
delta = 4.0
omega_R = 3.5

k0 = 0.5
L = 2 * np.pi / k0
    
# figure options
transparent = True
dpi = 300
save_options = {
    "transparent": transparent,
    "dpi": dpi
}

# simulation settings
t0 = 0
t1 = args.sim_time / t_r
n_savesteps = args.n_savesteps
save_at = np.linspace(t0, t1, n_savesteps)
delta_t = save_at[1] - save_at[0]

# saving parameters

parameters_json = json.dumps(
    {
        "k0": k0,
        "n_momentum_points": n_momentum_points,
        "L": L,
        "n_particles": n_particles,
        "hbar": hbar,
        "m_Yb": m_Yb,
        "soc": {
            "k_r": k_r,
            "delta": delta,
            "omega_R": omega_R,
        },
        "simulation_settings": {
            "t0": t0,
            "t1": t1 * t_r,
            "n_savesteps": n_savesteps,
            "delta_t": delta_t,
            "seed": seed
        }
    },
    indent=4
)
print("simulation parameters:")
print(parameters_json)

with open(os.path.join(data_dir, "parameters.json"), "w+") as f:
    f.write(parameters_json)

hilb = hilbert.PBCBox1D(
    L=L,
    n_momentum_points=n_momentum_points,
    n_particles=n_particles
)

# initial state
ks = np.array(hilb.momentums)
momentum_modes = tuple(np.argsort(ks ** 2)[:n_particles].tolist())


print(f"constructing initial state")

initial_state = hilb.get_momentum_eigenstate(
    tuple((mode, 1) for mode in momentum_modes)
)

print("constructing operators...")

hilb_dim = hilb.space_dim
H_scipy = ybsoc.sparse_hamiltonian_scipy_csr(
    hilb,
    hbar=hbar,
    k_r=k_r,
    m_Yb=m_Yb,
    delta=delta,
    omega_R=omega_R,
    U=0.0
)
steps_for_expm = expm_steps_est(-1j * H_scipy * delta_t / hbar, norm_bound=1.0)
taylor_order = args.taylor_order

print("recommanded n_steps:", steps_for_expm)

# momentum_num_op_diag = jnp.array(hilb.momentum_num_op_diags())
# NH
momentum_num_op_diag = hilb.momentum_num_op_diags()
momentum_num_op_diag = jnp.array(momentum_num_op_diag)

@jax.jit
def psis_to_momenum_nums(psi_batched: Complex[Array, "hdim batch"]):
    probs = jnp.abs(psi_batched) ** 2
    expected_numbers = einsum(momentum_num_op_diag, probs, "c hdim, hdim batch -> batch c")
    up_mom, down_mom = rearrange(expected_numbers, "batch (mom spin) -> spin batch mom", spin=2)
    return up_mom, down_mom

scaled_exponent = jsparse.BCSR.from_scipy_sparse((-1j * delta_t / hbar) * H_scipy / steps_for_expm)

up_nums_momentum = np.zeros((n_savesteps, n_momentum_points))
down_nums_momentum = np.zeros((n_savesteps, n_momentum_points))

print("plotting initial state...")

k_tick_labels = generate_k_labels(n_momentum_points, k0)

fig_path = os.path.join(figs_dir, "initial_momentum_numbers.png")
up_mom, down_mom = hilb.momentum_expected_numbers(initial_state)
plot_numbers(
    up_mom, down_mom, 
    tick_labels=k_tick_labels, 
    title="Initial state $\\langle n(q_x) \\rangle$",
    save_path=fig_path,
    save_options=save_options
)
psi = initial_state
psis = np.zeros((n_savesteps, hilb_dim))
# time evolutions:
for save_step in tqdm.trange(n_savesteps):
    psis[save_step, :] = psi
    up_mom, down_mom = hilb.momentum_expected_numbers(psi)
    up_nums_momentum[save_step, :] = up_mom
    down_nums_momentum[save_step, :] = down_mom

    psi = apply_expm(
        scaled_exponent,
        psi,
        steps_for_expm,
        taylor_order
    )
    
    
total_up = np.sum(up_nums_momentum, axis=-1) # <
total_down = np.sum(down_nums_momentum, axis=-1) # <

fraction_up = total_up / n_particles
fraction_down = total_down / n_particles


np.save(os.path.join(data_dir, "t.npy"), save_at)

np.save(os.path.join(data_dir, "total_up.npy"), total_up)
np.save(os.path.join(data_dir, "total_down.npy"), total_down)

np.save(os.path.join(data_dir, "up_nums_momentum.npy"), up_nums_momentum)
np.save(os.path.join(data_dir, "down_nums_momentum.npy"), down_nums_momentum)

np.save(os.path.join(data_dir, "state_trajectory.npy"), psis)

# =================================================
#         time evolution of total number
# =================================================

# plot figures
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 9))

ts = save_at * t_r
ax1.plot(ts, fraction_up)
ax1.set_ylabel("$\\langle n_{\\uparrow} \\rangle$")
ax1.set_ylim(0, 1.1)
# ax1.legend()

ax2.plot(ts, fraction_down)
ax2.set_ylabel("$\\langle n_{\\downarrow} \\rangle$")
ax2.set_ylim(0, 1.1)
# ax2.legend()

ax3.plot(ts, fraction_up + fraction_down)
ax3.set_ylabel("$\\langle n_{\\uparrow} \\rangle + \\langle n_{\\downarrow} \\rangle$")
ax3.set_ylim(0, 1.1)
# ax3.legend()
ax3.set_xlabel("$t\\,(ms)$")

plt.tight_layout()
fig_path = os.path.join(figs_dir, "total_number_time_evolution.png")
plt.savefig(fig_path, **save_options)
plt.show()

# =================================================
#         polarization at q_x = k_r
# =================================================

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

assert n_momentum_points % 2 == 1
center_idx = n_momentum_points // 2

idx = round(center_idx + 1 / k0)

k_r_up = up_nums_momentum[:, idx]
k_r_down = down_nums_momentum[:, idx]

spin_polarization = (k_r_up - k_r_down) / (k_r_up + k_r_down)

ts = save_at * t_r
ax.plot(ts, spin_polarization)
ax.set_ylabel("$(\\langle n_\\uparrow\\rangle - \\langle n_\\downarrow\\rangle)/(\\langle n_\\uparrow\\rangle + \\langle n_\\downarrow\\rangle)$")
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("$t\\,(ms)$")

plt.tight_layout()
fig_path = os.path.join(figs_dir, "spin_polarization_time_evolution.png")
plt.savefig(fig_path, **save_options)
plt.show()