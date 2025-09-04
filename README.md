





if cuSPARSE error occurs:
```bash
unset LD_LIBRARY_PATH
```


example:
```bash
uv run .\src\lindblad_qjm_baseline.py --run_name "test_{default}" --n_momentum_points 11 --n_particles 4
uv run .\src\fig4-eachtraj.py --run_name "trend_{default}" --n_momentum_points 13 --n_particles 7 --n_savesteps 61 --gamma 0.4 --T2 0.6 --seed 0
```