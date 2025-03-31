# AM-AFM
Toolbox for AM-AFM experiment data processing and force reconstruction.

for Python 3.12


## Requirements
install fro requirements-file:
```shell
pip install -r amafm/setup/requirements.txt
```


## Usage
### Curve selection
* view preprocessed curves in a GUI and select suitable measurements
* indices of the selected curves (according to their file-path) are stored in a pickle-file when finishing/closing the tool.
```shell
python select_experiments.py
```

### Preprocess experiment data
* load data from igor-binarywave files
* smooth measurements to reduce noise
* scale measurements using min-max-scaling
* align measurements on x- and y-axis
* store the data in `Measurement`-objects containing distance-, amplitude- and phase-data for approach and retraction of an experiment.
* see the methods doc-string for options
```python
from amafm import preprocessing
preprocessed_measurements, calibration_parameters = preprocessing.preprocess(data_directory)
```

### Create an average representative measurement from experiment data
* bin-wise average each type of measurment from all experiements
```python
from amafm import analysis
average_measurement = analysis.average_curves(preprocessed_measurements)
```

* also retrieve uncertainty of average measurements for multiple $\sigma$-distances
```python
average_measurement, measurments_at_zscores = analysis.average_curves(preprocessed_measurements, zscores=[1, 2, -1, -2])
```

### Reconstruct force curves
* Retrieve the force curve from a measurement
```python
from amafm import force
force_curve = force.force_hoelscher(average_measurement, calibration_parameters)
```
