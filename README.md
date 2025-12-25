
# `p_detection`

`p_detection` is a comprehensive Python package for detecting **P-waves** and **T-waves** from ECG signals using interpretable, rule-based, and template-matching techniques. It includes pathology filtering (AFIB, PVC, pauses), template clustering, phasor-based peak detection, dissociated P-wave logic, and performance evaluation using WFDB annotations.

## Features

- Preprocessing with adaptive filtering and median baseline removal  
- Template generation using Pearson correlation  
- Phasor transformation for peak enhancement  
- Detection of dissociated P-waves (P_dis)  
- Evaluation with MIT-BIH, BUTDB, QTDB AND LUDB using WFDB annotations  
- Visualization tools for template and peak analysis

## Installation

Install directly using pip:

```bash
pip install p_detection
```

Or install locally:

```bash

cd p_detection
pip install -e .
```

## Usage

Here's a basic example to detect P-waves and analyse for databases:

```python

from p_detection import ludb, but, bih, qtdb
import os


folder_path = 'C:/Users/InternPC6/Downloads/BUT_new'
folder_path1 = 'C:/Users/InternPC6/Downloads/LUDB_new'
folder_path2 = 'C:/Users/InternPC6/Downloads/MIT_p_wave'
folder_path3 = 'C:/Users/InternPC6/Downloads/QTDB'


p = 0.8

ludb(folder_path=folder_path1, p_co=p)
bih(folder_path=folder_path2, p_co=p)
qtdb(folder_path=folder_path3, p_co=p)
but(folder_path=folder_path, p_co=p)
```

- separate output from the device but need to use detect_p() separately like below

Here's a basic example to detect P-waves and analyse for databases:

```python


from p_detection import detect_p
import os


folder_path = 'path to data'
record= ['name of the records']


p = 0.8

det_p,data=detect_p(folder_path=folder_path,record_name=record)

```

-here is an example to import the zip file to googe colab and run on that
```python colab
from google.colab import files
uploaded = files.upload()  # Choose your zip files

!pip install p_detection.zip

from p_detection.p_detection import bih,but,ludb,qtdb
from google.colab import drive
drive.mount('/content/drive')


folder_path = '/content/drive/MyDrive/BUT_new'
folder_path1 = '/content/drive/MyDrive/LUDB_new'
folder_path2 = '/content/drive/MyDrive/MIT_p_wave'
folder_path3 = '/content/drive/MyDrive/QTDB'


p = 0.8

ludb(folder_path=folder_path1, p_co=p,plot=True)
qtdb(folder_path=folder_path3, p_co=p,plot=True,begin=600,end=630)
but(folder_path=folder_path, p_co=p,plot=True,begin=30,end=90)
bih(folder_path=folder_path2, p_co=p)
```

## Functions

- `detect_p(...)` – Full detection pipeline
- `acc_check_wfdb_ann(...)` – Evaluate against PhysioNet WFDB annotations
- `algorithm(...)` – Core signal processing + peak detection logic
- `load_ann(...)` – HDF5 signal and/or annotation from the device
- `plot_real_peaks_in_ecg_new(...)` – ECG visualizer with overlays
- `(dataset_name)(...)` - final wrapper for each dataset

## Datasets

Compatible with:
- MIT-BIH Arrhythmia Database (`mitbih`)
- LUDB 12 LEAD Database (`LUdb`)
- Brno University of Technology ECG Database (`butdb`)
- QTDB (`qtdb`)

  

## Credits

Created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) using the [`py-pkgs-cookiecutter`](https://github.com/py-pkgs/py-pkgs-cookiecutter) template.
