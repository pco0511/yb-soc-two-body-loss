





if cuSPARSE error occurs:
```bash
unset LD_LIBRARY_PATH
```


example:
```bash
uv run .\src\lindblad_qjm_baseline.py --run_name "test_{default}" --n_momentum_points 11 --n_particles 4
```