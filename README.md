# AM-AFM
Toolbox for AM-AFM experiment data processing and force reconstruction.

for Python 3.12


## Requirements
install from requirements-file:
```shell
pip install -r setup/requirements.txt
```


## Command line tools
### select_experiments.py
```shell
usage: python select_experiments.py data_dir [-h] [-t TARGET] [-n N_FILES] [-r REVISE] 

Open a GUI to screen curves and accept or reject measurements.  
The classification of the measurements is saved to `screened_files.csv` 
inside the given directory.

positional arguments:
  data_dir              Path to directory containing experiment `.ibw`-files to load.

options:
  -h, --help            show this help message and exit
  -t TARGET, --target TARGET
                        Target number of acceptable curves (default: 100).
  -n N_FILES, --n_files N_FILES
                        Number of unseen files to load and preprocess 
                        (default: -1, meaning all files).
  -r REVISE, --revise REVISE
                        If a previously saved `screened_files.csv` exists and this is given, 
                        these files are shown again to revisit the labels, 
                        otherwise they are skipped.
```

### force_tool.py
```shell
usage: python force_tool.py experiments_dir [-h] [-t TARGET] [-f FORCE_STEP_SIZE] [-m MODEL_PATH] [-s {phase,amp}] [-d {in,out}]

Load files of an experiment, select acceptable measurements, preprocess, average and compute force curve. Resulting average curves and force curve are saved to the experiments-directory.

positional arguments:
  experiments_dir       Path to directory containing the experiments `.ibw`-files to load.

options:
  -h, --help            show this help message and exit
  -t TARGET, --target TARGET
                        Target number of accepted curves (default: 100).
  -f FORCE_STEP_SIZE, --force_step_size FORCE_STEP_SIZE
                        Step size for averaging in the force calculation (default: 10).
  -m MODEL_PATH, --model_path MODEL_PATH
                        Path to the machine learning model for classification (default: `models/SVM.joblib`).
  -d {in,out}, --direction {in,out}
                        Direction of the probe, either `in` or `out` (default: `out`).
```


## API
* documentation of all user functions can be found in `docs.html`
* view the `demo.ipynb`-notebook for a overview of all functionalities and a usual workflow.
* all functions have the most reasonable parameter-values set as default wherever possible.
