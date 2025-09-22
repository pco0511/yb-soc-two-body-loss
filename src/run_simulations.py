import subprocess
import tomllib
import argparse
import time

def format_time(seconds: float) -> str:
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{int(hours)}h {int(mins)}m {secs:.2f}s"
    elif mins > 0:
        return f"{int(mins)}m {secs:.2f}s"
    else:
        return f"{secs:.2f}s"

def make_command(
    *,
    python_file_name: str,
    group_name: str,
    run_name: str,
    nh: bool,
    initial_state: str,
    theta: float,
    phi: float,
    k0: float,
    n_momentum_points: int,
    n_particles: int,
    n_savesteps: int,
    n_trajectories: int,
    batch_size: int,
    jumps_per_step: int,
    gamma: float,
    T2: float,
    temperature: None | float,
    sim_time: float,
    seed: int
):
    commands = [
        "uv", "run", python_file_name, 
        "--group_name", group_name,
        "--run_name", run_name, 
        "--initial_state", initial_state,
        "--theta", f"{theta:.8f}",
        "--phi", f"{phi:.8f}",
        "--k0", f"{k0:.8f}",
        "--n_momentum_points", f"{n_momentum_points}", 
        "--n_particles", f"{n_particles}", 
        "--n_savesteps", f"{n_savesteps}",
        "--n_trajectories", f"{n_trajectories}",
        "--batch_size", f"{batch_size}",
        "--jumps_per_step", f"{jumps_per_step}",
        "--gamma", f"{gamma:.8f}",
        "--T2", f"{T2:.8f}",
        "--sim_time", f"{sim_time:.8f}",
        "--seed", f"{seed}"
    ]
    if nh:
        commands.append("--nh")
    if temperature is not None:
        commands.extend(["--temperature", f"{temperature:.4f}"])
    return commands

def main():
    parser = argparse.ArgumentParser(description="Run simulations from a TOML configuration file.")
    parser.add_argument("--config", help="Path to the TOML configuration file.")
    args = parser.parse_args()
    
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    default_params = config.get('default', {})
    simulations = config.get('simulations', [])
    
    print(f"running total {len(simulations)} simulations.")

    total_start_time = time.monotonic()
    
    for i, sim_specific_params in enumerate(simulations):
        print(f"\n---  {i+1}/{len(simulations)}  ---")
        
        final_params = default_params.copy()
        final_params.update(sim_specific_params)
        
        print("   [parameters]")
        for key, val in sim_specific_params.items():
            print(f"     - {key}: {val} (overrided)")
        
        sim_start_time = time.monotonic()
        
        try:
            comm = make_command(**final_params)
            subprocess.run(comm, check=True)
            
            sim_end_time = time.monotonic()
            sim_duration = sim_end_time - sim_start_time
            
            print(f"--- simulation {i+1} succeed (took {format_time(sim_duration)}) ---")
        except TypeError as e:
            print(f"--- configuration error ---")
            print(e)
            break
        except subprocess.CalledProcessError as e:
            sim_end_time = time.monotonic()
            sim_duration = sim_end_time - sim_start_time
            print(f"--- Simulation {i+1} failed (after {format_time(sim_duration)}) ---")
            print(e)
            break
    
    total_end_time = time.monotonic()
    total_duration = total_end_time - total_start_time
    print("\n==========================================")
    print(f"All simulations finished. Total time: {format_time(total_duration)}")
    print("==========================================")

if __name__ == "__main__":
    main()