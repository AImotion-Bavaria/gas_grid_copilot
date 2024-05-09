# Gas Grid Copilot Demo
## Setup hints:

The main file of interest is `ikigas_dashboard_demo.py`

Use a conda/venv environment with `requirements.txt` or `energy.yml` (or `energy_linux.yml`)

Run `ikigas_dashboard_demo.py` 
1. Once with `prepare_and_train = True` (in the main method)
2. You should now have a directory `cached_models` containing a number of .zip files for the trained agents and a pickled file of state plots
3. Run `ikigas_dashboard_demo.py` again with  `prepare_and_train = False`

## Activating Pyqualicision
If you have `pyqualicision`binaries, you may use `demo_plot = IKIGasDashboard(use_qualicision=True)` in `ikigas_dashboard_demo.py` otherwise use `demo_plot = IKIGasDashboard(use_qualicision=False)` to default to simple reward correlations.
