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
from quant_mech.qjm import qjm_step
import argparse

print(f"Using: {jax.devices()}")


parser = argparse.ArgumentParser(description="Lindblad QJM Simulation for Yb SOC System")

parser.add_argument('--group_name', type=str, default="temp")
parser.add_argument('--run_name', type=str, help='Root directory for data and figures')
parser.add_argument('--nh', action='store_true', help='Use Non-Hermitian Hamiltonian instead of Lindblad operators')
parser.add_argument('--initial_state', type=str, required=True, help='initial state for simulation.', choices=[
    'aligned_up', 'mixture', 'superposition', 'soc_ground', 'custom'
])
parser.add_argument('--theta', type=float, default=0.0, help='parameter for super position: \\cos(\\theta/2)\\ket{0} + e^{i\\phi}\\sin(\\theta/2)\\ket{1}')
parser.add_argument('--phi', type=float, default=0.0, help='parameter for super position: \\cos(\\theta/2)\\ket{0} + e^{i\\phi}\\sin(\\theta/2)\\ket{1}')
parser.add_argument('--n_momentum_points', type=int, required=True, help='Number of momentum points')
parser.add_argument('--n_particles', type=int, required=True, help='Number of particles')
parser.add_argument('--n_savesteps', type=int, default=51, help='Number of time steps to save')
parser.add_argument('--n_trajectories', type=int, default=512, help='Number of QJM trajectories')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for trajectories')
parser.add_argument('--jumps_per_step', type=int, default=4, help='Number of jumps per saved time step')
parser.add_argument('--gamma', type=float, default=0.4, help='2-body dissipation rate')
parser.add_argument('--T2', type=float, default=1.0, help='T2 decoherence')
parser.add_argument('--temperature', type=float, help="Temperature in micro Kelvin. Only works in initial_state=soc_ground. If not given, it is treated as zero temperature.")
parser.add_argument('--sim_time', type=float, default=1.0, help='simulation time in mili second')
parser.add_argument('--seed', type=int, help='Random seed if not given, the timestamp will be used.')

args = parser.parse_args()

# Generate timestamp (MMDDHHMM)
CURRENT_TIME_STAMP = time.strftime("%m%d%H%M")

DATA_ROOT = Path(os.path.dirname(__file__)).parent / "data" / args.group_name

if args.nh:
    DEFAULT_NAME = f"nh_{args.n_momentum_points}_{args.n_particles}_{CURRENT_TIME_STAMP}"
else:
    DEFAULT_NAME = f"lindblad_{args.n_momentum_points}_{args.n_particles}_{CURRENT_TIME_STAMP}"
        
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

NH = args.nh
if args.seed is not None:
    seed = args.seed
else:
    seed = int(time.time())
    
key = jax.random.key(seed)

os.makedirs(data_dir, exist_ok=True)
os.makedirs(figs_dir, exist_ok=True)

if NH:
    print(f"NH mode")
else:
    print(f"Lindblad mode")

os.makedirs(DATA_ROOT, exist_ok=True)

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
gamma = args.gamma

T2 = args.T2 / t_r

if args.temperature is not None:
    temperature = args.temperature / T_r
    zerotemp = False
else:
    temperature = 0.0
    zerotemp = True
    
# figure options
transparent = True
dpi = 300
save_options = {
    "transparent": transparent,
    "dpi": dpi
}

# simulation settings
t0 = 0
t1 = args.sim_time / 0.1129
n_savesteps = args.n_savesteps
save_at = np.linspace(t0, t1, n_savesteps)
step_size = save_at[1] - save_at[0]

n_trajectories = args.n_trajectories
batch_size = args.batch_size
jumps_per_step = args.jumps_per_step

assert n_trajectories % batch_size == 0

num_batches = n_trajectories // batch_size
total_n_steps = (n_savesteps - 1) * jumps_per_step
delta_t = (t1 - t0) / total_n_steps

# batch_size = args.batch_size
# jumps_per_step = args.jumps_per_step
# n_savesteps = args.n_savesteps
# n_trajectories = args.n_trajectories

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
        "pa": {
            "gamma": gamma,
            "T2": T2 * t_r
        },
        "simulation_settings": {
            "t0": t0,
            "t1": t1 * t_r,
            "mode": 'non-Hermitian' if NH else "lindbladian",
            "zerotemp": zerotemp,
            "temperature": temperature,
            "n_savesteps": n_savesteps,
            "step_size": step_size,
            "n_trajectories": n_trajectories,
            "batch_size": batch_size,
            "jumps_per_step": jumps_per_step,
            "num_batches": num_batches,
            "total_n_steps": total_n_steps,
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
    n_particles=list(range(n_particles, -1, -2))
)
hilb.print_info()
max_par_sub_hilb = hilbert.PBCBox1D(
    L=L,
    n_momentum_points=n_momentum_points,
    n_particles=n_particles
)
subspace_dim = hilb.space_dim - max_par_sub_hilb.space_dim

# initial state
ks = np.array(hilb.momentums)
momentum_modes = tuple(np.argsort(ks ** 2)[:n_particles].tolist())


print(f"constructing initial state: {args.initial_state}")

match (args.initial_state):
    case "aligned_up":
        initial_state = hilb.get_momentum_eigenstate(
            tuple((mode, 1) for mode in momentum_modes)
        )
    case "mixture":
        initial_state = hilb.get_momentum_eigenstate(
            tuple((mode, (-1) ** idx) for idx, mode in enumerate(momentum_modes))
        )
    case "superposition":
        theta = args.theta
        phi = args.phi
        alpha = np.cos(theta / 2)
        beta = np.exp(1j * phi) * np.sin(theta / 2)
        spin_amps = {
            1: alpha,
            -1: beta
        }
        spins = [-1, 1]
        initial_state = hilb.zero_state()
        for spin_config in itertools.product(spins, repeat=n_particles):
            amp = np.prod([spin_amps[s] for s in spin_config])
            basis_config = tuple(
                (mode, spin) for mode, spin in zip(momentum_modes, spin_config)
            )
            basis = hilb.get_momentum_eigenstate(basis_config)
            initial_state = initial_state + amp * basis
    case "soc_ground":
        
        spectrum, lowlyings = ybsoc.manybody_spectrum(
            max_par_sub_hilb,
            hbar, k_r, m_Yb, delta, omega_R
        )
        
        if zerotemp:
            ss = next(iter(lowlyings))
            initial_state = hilb.from_single_states(ss)
            E_gs = spectrum[0]
            print(f"Ground state energy: {E_gs}")
        else:
            boltz = np.exp(-spectrum / temperature)
            Z = np.sum(boltz)
            boltz /= Z
            n_lows = 0
            for p in boltz:
                if p < 1e-4:
                    break
                n_lows += 1
            print(f"Using {n_lows} low-lying states for finite-temperature calculations.")
            initial_states = np.zeros((n_lows, hilb.space_dim), dtype=np.complex128)
            if initial_states.nbytes > 2 ** 32:
                print("Warning: initial_states takes more than 4 GB")
                
            if initial_states.nbytes > 32 * 2 ** 30:
                raise ValueError("candidate of initial state is too mush.")
            
            cnt = 0
            for ss in lowlyings:
                if cnt >= n_lows:
                    break
                try:
                    psi = hilb.from_single_states(ss)
                except Exception as e:
                    print(f"{ss=}")
                    raise e
                initial_states[cnt, :] = psi
                cnt += 1
            boltz_truncated = boltz[:n_lows]
            Z_truncated = np.sum(boltz_truncated)
            boltz_truncated /= Z_truncated
            boltz_logits = -spectrum[:n_lows] / temperature
    case "custom":
        if n_particles == 6:
            allowed_momentum_modes = [6, 7, 8, 9, 10, 11]
        elif n_particles == 7:
            allowed_momentum_modes = [5, 6, 7, 8, 9, 10, 11]
        else:
            raise ValueError(f"unsupported n_particles: {n_particles}")
        spectrum, lowlyings = ybsoc.manybody_spectrum(
            max_par_sub_hilb,
            hbar, k_r, m_Yb, delta, omega_R,
            allowed_momentum_modes, lower_only=True
        )
        ss = next(iter(lowlyings))
        initial_state = hilb.from_single_states(ss)
    case _:
        raise ValueError(f"invalid initial state type: {args.initial_state}")

# if zerotemp:

# if zerotemp:    
#     n_states = 1
# else:
#     n_states = 100
    
# E, lowlying = scipy.sparse.linalg.eigsh(
#     soc_hamiltonian,
#     k=n_states,
#     which='SA',
#     tol=1e-6,
#     return_eigenvectors=True
# )
# E_gs = E[0]
# psi_gs = lowlying[:, 0]

# eps = 1e-6

# print("prob ratio, exp((E - E_gs)/T):")
# [print](np.exp(-(E[-1] - E_gs) / temperature))
    
# @partial(jax.jit, static_argnames=['n_samples'])
# def sample_from_boltzmann(n_samples, lowlyings, boltzmann_logits, key):
#     indices = jax.random.categorical(key, boltzmann_logits, shape=(n_samples,))
#     return lowlyings[:, indices]

# operators
print("constructing operators...")

loss_channels = ybsoc.lindblad_two_body_loss_scipy_csr(
    hilb, gamma, reduced_rows=True
)
n_loss_channels = len(loss_channels)
stacked_loss_op_scipy = scipy.sparse.vstack(loss_channels, format='csr')

del loss_channels
gc.collect()

hilb_dim = hilb.space_dim
H_eff_scipy = ybsoc.sparse_hamiltonian_scipy_csr(
    hilb,
    hbar=hbar,
    k_r=0.0,
    m_Yb=m_Yb,
    delta=0.0,
    omega_R=0.0,
    U=-(1/2) * 1j * gamma * hbar
)
H_eff_scipy = H_eff_scipy - (1j * hbar) / (2 * T2) * scipy.sparse.identity(hilb_dim, format='csr')

steps_for_expm = expm_steps_est(-1j * H_eff_scipy * delta_t / hbar, norm_bound=1.0)
taylor_order = 8

print("recommanded n_steps:", steps_for_expm)

# momentum_num_op_diag = jnp.array(hilb.momentum_num_op_diags())
# NH
momentum_num_op_diag = hilb.momentum_num_op_diags()
momentum_num_op_diag_nh = momentum_num_op_diag.copy()
momentum_num_op_diag_nh[:, :subspace_dim] = 0

momentum_num_op_diag = jnp.array(momentum_num_op_diag)
momentum_num_op_diag_nh = jnp.array(momentum_num_op_diag_nh)


@jax.jit
def psis_to_momenum_nums(psi_batched: Complex[Array, "hdim batch"]):
    probs = jnp.abs(psi_batched) ** 2
    expected_numbers = einsum(momentum_num_op_diag, probs, "c hdim, hdim batch -> batch c")
    up_mom, down_mom = rearrange(expected_numbers, "batch (mom spin) -> spin batch mom", spin=2)
    return up_mom, down_mom

@jax.jit
def psis_to_momenum_nums_nh(psi_batched: Complex[Array, "hdim batch"]):
    probs = jnp.abs(psi_batched) ** 2
    expected_numbers = einsum(momentum_num_op_diag, probs, "c hdim, hdim batch -> batch c")
    up_mom, down_mom = rearrange(expected_numbers, "batch (mom spin) -> spin batch mom", spin=2)
    expected_numbers_nh = einsum(momentum_num_op_diag_nh, probs, "c hdim, hdim batch -> batch c")
    up_mom_nh, down_mom_nh = rearrange(expected_numbers_nh, "batch (mom spin) -> spin batch mom", spin=2)
    return up_mom, down_mom, up_mom_nh, down_mom_nh


scaled_exponent = jsparse.BCSR.from_scipy_sparse((-1j * delta_t / hbar) * H_eff_scipy / steps_for_expm)
stacked_loss_op = jsparse.BCSR.from_scipy_sparse(stacked_loss_op_scipy)

up_nums_momentum = np.zeros((n_trajectories, n_savesteps, n_momentum_points))
down_nums_momentum = np.zeros((n_trajectories, n_savesteps, n_momentum_points))
if NH:
    up_nums_momentum_nh = np.zeros((n_trajectories, n_savesteps, n_momentum_points))
    down_nums_momentum_nh = np.zeros((n_trajectories, n_savesteps, n_momentum_points))

n_position_points = n_momentum_points
up_nums_position = np.zeros((n_trajectories, n_savesteps, n_position_points))
down_nums_position = np.zeros((n_trajectories, n_savesteps, n_position_points))
if NH:
    up_nums_position_nh = np.zeros((n_trajectories, n_savesteps, n_position_points))
    down_nums_position_nh = np.zeros((n_trajectories, n_savesteps, n_position_points))

if NH:
    state_norm_square_nh = np.zeros((n_trajectories, n_savesteps))
    state_norm_square_nh[:, 0] = 1


if zerotemp:
    up_mom, down_mom = hilb.momentum_expected_numbers(initial_state)
    up_pos, down_pos = hilb.position_expected_numbers(initial_state)
else:
    updown_moms = hilb.momentum_expected_numbers(initial_states)
    updown_poss = hilb.position_expected_numbers(initial_states)

    up_moms = updown_moms[:, 0, :]
    down_moms = updown_moms[:, 1, :]
    up_poss = updown_poss[:, 0, :]
    down_poss = updown_poss[:, 1, :]
    
    up_mom = np.einsum("i,im->m", boltz_truncated, up_moms)
    down_mom = np.einsum("i,im->m", boltz_truncated, down_moms)
    up_pos = np.einsum("i,im->m", boltz_truncated, up_poss)
    down_pos = np.einsum("i,im->m", boltz_truncated, down_poss)
    
up_nums_momentum[:, 0, :] = up_mom[np.newaxis, :]
down_nums_momentum[:, 0, :] = down_mom[np.newaxis, :]
if NH:
    up_nums_momentum_nh[:, 0, :] = up_mom[np.newaxis, :]
    down_nums_momentum_nh[:, 0, :] = down_mom[np.newaxis, :]

up_nums_position[:, 0, :] = up_pos[np.newaxis, :]
down_nums_position[:, 0, :] = down_pos[np.newaxis, :]
if NH:
    up_nums_position_nh[:, 0, :] = up_pos[np.newaxis, :]
    down_nums_position_nh[:, 0, :] = down_pos[np.newaxis, :]


print("plotting initial state...")

k_tick_labels = generate_k_labels(n_momentum_points, k0)

fig_path = os.path.join(figs_dir, "initial_momentum_numbers.png")
plot_numbers(
    up_mom, down_mom, 
    tick_labels=k_tick_labels, 
    title="Initial state $\\langle n(q_x) \\rangle$",
    save_path=fig_path,
    save_options=save_options
)

fig_path = os.path.join(figs_dir, "initial_position_numbers.png")
plot_numbers(
    up_pos, down_pos, 
    title="Initial State $\\langle n(x) \\rangle$",
    save_path=fig_path,
    save_options=save_options
)

cpu_device = jax.devices('cpu')[0]
gpu_device = jax.devices('gpu')[0]


if not zerotemp:
    initial_states_jaxcpu = jnp.array(initial_states, device=cpu_device)
    boltz_logits_jaxcpu = jnp.array(boltz_logits, device=cpu_device)

    @partial(jax.jit, static_argnums=(3))
    def sample_from_boltzmann(key, initial_states, boltz_logits, batch_size):
        indices = jax.random.categorical(key, boltz_logits, shape=(batch_size,))
        psis = initial_states[indices]
        psis = rearrange(psis, "b n -> n b")
        return jax.device_put(psis, device=gpu_device)

# qjm iterations
for i_batch in range(num_batches):
    if zerotemp:
        psi_batched = np.repeat(initial_state[:, np.newaxis], batch_size, axis=1) # prepare batched states
        psi_batched = jnp.array(psi_batched, dtype=np.complex128) # convert to jax array
    else:
        key, subkey = jax.random.split(key)
        psi_batched = sample_from_boltzmann(subkey, initial_states_jaxcpu, boltz_logits_jaxcpu, batch_size)

    for iteration in tqdm.trange(total_n_steps, desc=f"batch {i_batch}: {(i_batch+1) * batch_size}/{n_trajectories} Trajectories", leave=False):
        # start = time.time()
        key, subkey = jax.random.split(key)
        psi_batched = qjm_step(
            psi_batched,
            scaled_exponent,
            stacked_loss_op,
            delta_t=delta_t,
            hbar=hbar,
            n_expm_steps=steps_for_expm,
            taylor_order=taylor_order,
            n_loss_channels=n_loss_channels,
            loss_op_rank=subspace_dim,
            T2=T2,
            key=subkey  
        )
        # psi_batched.block_until_ready()
        # print("qjm step", time.time() - start)

        # start = time.time()
        # calculate obaservables
        if (iteration + 1) % jumps_per_step != 0:
            # print("obeservables 1", time.time() - start)
            continue

        i = (iteration + 1) // jumps_per_step

        # print("stamp 1", time.time() - start)
        if NH:
            up_num_mom, down_num_mom, up_num_mom_nh, down_num_mom_nh = psis_to_momenum_nums_nh(psi_batched)
        else:
            up_num_mom, down_num_mom = psis_to_momenum_nums(psi_batched)
        # up_num_sum.block_until_ready()
        # down_num_sum.block_until_ready()
        # print("stamp 2", time.time() - start)

        batch_start = i_batch * batch_size
        batch_end = (i_batch + 1) * batch_size

        psis = np.asarray(psi_batched).T # (n, b) => (b, n)

        up_nums_momentum[batch_start:batch_end, i, :] = np.asarray(up_num_mom)
        down_nums_momentum[batch_start:batch_end, i, :] = np.asarray(down_num_mom)
        if NH:
            up_nums_momentum_nh[batch_start:batch_end, i, :] = np.asarray(up_num_mom_nh)
            down_nums_momentum_nh[batch_start:batch_end, i, :] = np.asarray(down_num_mom_nh)
            state_norm_square_nh[batch_start:batch_end, i] = np.linalg.norm(psis[:, subspace_dim:], axis=1) ** 2
        # print("stamp 3", time.time() - start)
        
        if i < n_savesteps - 1:
            # print("obeservables 2", time.time() - start)
            continue

        nums = hilb.position_expected_numbers(psis)
        up_pos, down_pos = rearrange(nums, "b s x -> s b x")
        if NH:
            nums_nh = hilb.position_expected_numbers(psis, nh=True)
            up_pos_nh, down_pos_nh = rearrange(nums_nh, "b s x -> s b x")
        
        up_nums_position[batch_start:batch_end, i, :] += np.asarray(up_pos)
        down_nums_position[batch_start:batch_end, i, :] += np.asarray(down_pos)

        # print("obeservables 3", time.time() - start)
        
up_nums_momentum_mean = np.mean(up_nums_momentum, axis=0)
up_nums_momentum_std = np.std(up_nums_momentum, axis=0)

down_nums_momentum_mean = np.mean(down_nums_momentum, axis=0)
down_nums_momentum_std = np.std(down_nums_momentum, axis=0)

up_nums_position_mean = np.mean(up_nums_position, axis=0)
up_nums_position_std = np.std(up_nums_position, axis=0)

down_nums_position_mean = np.mean(down_nums_position, axis=0)
down_nums_position_std = np.mean(down_nums_position, axis=0)

if NH:
    up_nums_momentum_mean_nh = np.mean(up_nums_momentum_nh, axis=0) # <
    up_nums_momentum_std_nh = np.std(up_nums_momentum_nh, axis=0) # <
    
    down_nums_momentum_mean_nh = np.mean(down_nums_momentum_nh, axis=0) # <
    down_nums_momentum_std_nh = np.std(down_nums_momentum_nh, axis=0) # <

    up_nums_position_mean_nh = np.mean(up_nums_position_nh, axis=0)
    up_nums_position_std_nh = np.std(up_nums_position_nh, axis=0)

    down_nums_position_mean_nh = np.mean(down_nums_position_nh, axis=0)
    down_nums_position_std_nh = np.mean(down_nums_position_nh, axis=0)

total_up = np.sum(up_nums_momentum_mean, axis=-1) # <
total_down = np.sum(down_nums_momentum_mean, axis=-1) # <

if NH:
    total_up_nh = np.sum(up_nums_momentum_mean_nh, axis=-1) # <
    total_down_nh = np.sum(down_nums_momentum_mean_nh, axis=-1) # <

fraction_up = total_up / n_particles
fraction_down = total_down / n_particles

if NH:
    fraction_up_nh = total_up_nh / n_particles # <
    fraction_down_nh = total_down_nh / n_particles # <


np.save(os.path.join(data_dir, "t.npy"), save_at)

np.save(os.path.join(data_dir, "total_up.npy"), total_up)
np.save(os.path.join(data_dir, "total_down.npy"), total_down)

np.save(os.path.join(data_dir, "total_up_nh.npy"), total_up_nh)
np.save(os.path.join(data_dir, "total_down_nh.npy"), total_down_nh)

np.save(os.path.join(data_dir, "full_up_nums_momentum.npy"), up_nums_momentum)
np.save(os.path.join(data_dir, "full_down_nums_momentum.npy"), down_nums_momentum)

np.save(os.path.join(data_dir, "full_up_nums_momentum_nh.npy"), up_nums_momentum_nh)
np.save(os.path.join(data_dir, "full_down_nums_momentum_nh.npy"), down_nums_momentum_nh)
    

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


# ==================================================
#    non-Hermitian time evolution of total number   
# ==================================================


# plot figures
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 9))

ts = save_at * t_r
ax1.plot(ts, fraction_up_nh)
ax1.set_ylabel("$\\langle n_{\\uparrow} \\rangle$")
ax1.set_ylim(0, 1.1)
# ax1.legend()

ax2.plot(ts, fraction_down_nh)
ax2.set_ylabel("$\\langle n_{\\downarrow} \\rangle$")
ax2.set_ylim(0, 1.1)
# ax2.legend()

ax3.plot(ts, fraction_up_nh + fraction_down_nh)
ax3.set_ylabel("$\\langle n_{\\uparrow} \\rangle + \\langle n_{\\downarrow} \\rangle$")
ax3.set_ylim(0, 1.1)
# ax3.legend()
ax3.set_xlabel("$t\\,(ms)$")

plt.tight_layout()
fig_path = os.path.join(figs_dir, "total_number_time_evolution_nh.png")
plt.savefig(fig_path, **save_options)
plt.show()


# ==== state norm ==========================================

# plot figures
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ts = save_at * t_r
ax.plot(ts, np.mean(state_norm_square_nh, axis=0), label='square norm')
ax.legend()
ax.set_ylim(0, 1.1)
ax.set_xlabel("$t\\,(ms)$")

plt.tight_layout()
fig_path = os.path.join(figs_dir, "square_norm_nh.png")
plt.savefig(fig_path, **save_options)
plt.show()
  
# ==========================================================

final_position_nums_up = up_nums_position_mean[-1]
final_position_nums_up_fluctuation = up_nums_position_std[-1]

final_position_nums_down = down_nums_position_mean[-1]
final_position_nums_down_fluctuation = down_nums_position_std[-1]

final_momentum_nums_up = up_nums_momentum_mean[-1]
final_momentum_nums_up_fluctuation = up_nums_momentum_std[-1]

final_momentum_nums_down = down_nums_momentum_mean[-1]
final_momentum_nums_down_fluctuation = down_nums_momentum_std[-1]

np.save(os.path.join(data_dir, "final_position_numbers_up.npy"), final_position_nums_up)
np.save(os.path.join(data_dir, "final_position_numbers_up_fluctuation.npy"), final_position_nums_up_fluctuation)

np.save(os.path.join(data_dir, "final_position_numbers_down.npy"), final_position_nums_down)
np.save(os.path.join(data_dir, "final_position_numbers_down_fluctuation.npy"), final_position_nums_down_fluctuation)

np.save(os.path.join(data_dir, "total_momentum_numbers_up.npy"), final_momentum_nums_up)
np.save(os.path.join(data_dir, "total_momentum_numbers_up_fluctuation.npy"), final_momentum_nums_up_fluctuation)

np.save(os.path.join(data_dir, "total_momentum_numbers_down.npy"), final_momentum_nums_down)
np.save(os.path.join(data_dir, "total_momentum_numbers_down_fluctuation.npy"), final_momentum_nums_down_fluctuation)


# Non Hermitian case =======================================
if NH:
    final_position_nums_up_nh = up_nums_position_mean_nh[-1]
    final_position_nums_up_fluctuation_nh = up_nums_position_std_nh[-1]

    final_position_nums_down_nh = down_nums_position_mean_nh[-1]
    final_position_nums_down_fluctuation_nh = down_nums_position_std_nh[-1]

    final_momentum_nums_up_nh = up_nums_momentum_mean_nh[-1]
    final_momentum_nums_up_fluctuation_nh = up_nums_momentum_std_nh[-1]

    final_momentum_nums_down_nh = down_nums_momentum_mean_nh[-1]
    final_momentum_nums_down_fluctuation_nh = down_nums_momentum_std_nh[-1]

    np.save(os.path.join(data_dir, "final_position_numbers_up_nh.npy"), final_position_nums_up_nh)
    np.save(os.path.join(data_dir, "final_position_numbers_up_fluctuation_nh.npy"), final_position_nums_up_fluctuation_nh)

    np.save(os.path.join(data_dir, "final_position_numbers_down_nh.npy"), final_position_nums_down_nh)
    np.save(os.path.join(data_dir, "final_position_numbers_down_fluctuation_nh.npy"), final_position_nums_down_fluctuation_nh)

    np.save(os.path.join(data_dir, "total_momentum_numbers_up_nh.npy"), final_momentum_nums_up_nh)
    np.save(os.path.join(data_dir, "total_momentum_numbers_up_fluctuation_nh.npy"), final_momentum_nums_up_fluctuation_nh)

    np.save(os.path.join(data_dir, "total_momentum_numbers_down_nh.npy"), final_momentum_nums_down_nh)
    np.save(os.path.join(data_dir, "total_momentum_numbers_down_fluctuation_nh.npy"), final_momentum_nums_down_fluctuation_nh)

    np.save(os.path.join(data_dir, "state_norm_square_nh.npy"), state_norm_square_nh)
    
# ==========================================================


fig_path = os.path.join(figs_dir, "final_position_numbers.png")
plot_numbers(
    final_position_nums_up,
    final_position_nums_down,
    title="Final Position Space Numbers",
    save_path=fig_path,
    save_options=save_options
)
fig_path = os.path.join(figs_dir, "final_momentum_numbers.png")
plot_numbers(
    final_momentum_nums_up,
    final_momentum_nums_down,
    tick_labels=k_tick_labels,
    title="Final Momentum Space Numbers",
    save_path=fig_path,
    save_options=save_options
)    
    
if NH:
    fig_path = os.path.join(figs_dir, "final_position_numbers_nh.png")
    plot_numbers(
        final_position_nums_up_nh,
        final_position_nums_down_nh,
        title="Final Position Space Numbers (Non-Hermitian)",
        save_path=fig_path,
        save_options=save_options
    )
    fig_path = os.path.join(figs_dir, "final_momentum_numbers_nh.png")
    plot_numbers(
        final_momentum_nums_up_nh,
        final_momentum_nums_down_nh,
        tick_labels=k_tick_labels,
        title="Final Momentum Space Numbers (Non-Hermitian)",
        save_path=fig_path,
        save_options=save_options
    )


# ================================================
#               loss fractions
# ================================================

def calc_loss_frac(initial, final, eps=1e-6):
    return np.where(np.abs(initial) < eps, np.zeros_like(initial), 1 - final / initial)

initial_position_nums_up = up_nums_position_mean[0]
initial_position_nums_down = down_nums_position_mean[0]
initial_momentum_nums_up = up_nums_momentum_mean[0]
initial_momentum_nums_down = down_nums_momentum_mean[0]

initial_position_nums_total = initial_position_nums_up + initial_position_nums_down
initial_momentum_nums_total = initial_momentum_nums_up + initial_momentum_nums_down
final_position_nums_total = final_position_nums_up + final_position_nums_down
final_momentum_nums_total = final_momentum_nums_up + final_momentum_nums_down

loss_fraction_up_position = calc_loss_frac(initial_position_nums_up, final_position_nums_up)
loss_fraction_down_position = calc_loss_frac(initial_position_nums_down, final_position_nums_down)
loss_fraction_total_position = calc_loss_frac(initial_position_nums_total, final_position_nums_total)

loss_fraction_up_momentum = calc_loss_frac(initial_momentum_nums_up, final_momentum_nums_up)
loss_fraction_down_momentum = calc_loss_frac(initial_momentum_nums_down, final_momentum_nums_down)
loss_fraction_total_momentum = calc_loss_frac(initial_momentum_nums_total, final_momentum_nums_total)


fig_path = os.path.join(figs_dir, "position_space_loss_fraction.png")
plot_fractions(
    loss_fraction_up_position,
    loss_fraction_down_position,
    loss_fraction_total_position,
    title="Position Space Loss Fractions",
    save_path=fig_path,
    save_options=save_options
)

fig_path = os.path.join(figs_dir, "momentum_space_loss_fraction.png")
plot_fractions(
    loss_fraction_up_momentum,
    loss_fraction_down_momentum,
    loss_fraction_total_momentum,
    tick_labels=k_tick_labels,
    title="Momentum Space Loss Fractions",
    save_path=fig_path,
    save_options=save_options
)

if NH:
    initial_position_nums_up_nh = up_nums_position_mean_nh[0]
    initial_position_nums_down_nh = down_nums_position_mean_nh[0]
    initial_momentum_nums_up_nh = up_nums_momentum_mean_nh[0]
    initial_momentum_nums_down_nh = down_nums_momentum_mean_nh[0]

    initial_position_nums_total_nh = initial_position_nums_up_nh + initial_position_nums_down_nh
    initial_momentum_nums_total_nh = initial_momentum_nums_up_nh + initial_momentum_nums_down_nh
    final_position_nums_total_nh = final_position_nums_up_nh + final_position_nums_down_nh
    final_momentum_nums_total_nh = final_momentum_nums_up_nh + final_momentum_nums_down_nh

    loss_fraction_up_position_nh = calc_loss_frac(initial_position_nums_up_nh, final_position_nums_up_nh)
    loss_fraction_down_position_nh = calc_loss_frac(initial_position_nums_down_nh, final_position_nums_down_nh)
    loss_fraction_total_position_nh = calc_loss_frac(initial_position_nums_total_nh, final_position_nums_total_nh)

    loss_fraction_up_momentum_nh = calc_loss_frac(initial_momentum_nums_up_nh, final_momentum_nums_up_nh)
    loss_fraction_down_momentum_nh = calc_loss_frac(initial_momentum_nums_down_nh, final_momentum_nums_down_nh)
    loss_fraction_total_momentum_nh = calc_loss_frac(initial_momentum_nums_total_nh, final_momentum_nums_total_nh)


    fig_path = os.path.join(figs_dir, "position_space_loss_fraction_nh.png")
    plot_fractions(
        loss_fraction_up_position_nh,
        loss_fraction_down_position_nh,
        loss_fraction_total_position_nh,
        title="Position Space Loss Fractions (Non-Hermitian)",
        save_path=fig_path,
        save_options=save_options
    )

    fig_path = os.path.join(figs_dir, "momentum_space_loss_fraction_nh.png")
    plot_fractions(
        loss_fraction_up_momentum_nh,
        loss_fraction_down_momentum_nh,
        loss_fraction_total_momentum_nh,
        tick_labels=k_tick_labels,
        title="Momentum Space Loss Fractions (Non-Hermitian)",
        save_path=fig_path,
        save_options=save_options
    )