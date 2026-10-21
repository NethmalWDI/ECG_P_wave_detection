
# `p_detection`

`p_detection` is a comprehensive Python package for detecting **P-waves** and **T-waves** from ECG signals using interpretable, rule-based, and template-matching techniques. It includes pathology filtering (AFIB, PVC, pauses), template clustering, phasor-based peak detection, dissociated P-wave logic, and performance evaluation using WFDB annotations.

Developed during a Research Engineer internship at **SYNERGEN Technology Labs** (Dec 2024 – Jul 2025). Full write-up: [report/Report.pdf](report/Report.pdf).

## Results

Evaluated against expert annotations on four public PhysioNet databases (final values from the report):

| Database | PPV (%) | Sensitivity (%) |
|---|---|---|
| MIT-BIH Arrhythmia (excluding record 207) | 96.1 | 95.4 |
| BUT P-wave Database (BUT PDB) | 89.6 | 89.1 |
| LUDB\* | 90.4 | 97.5 |
| QTDB | 90.4 | 93.2 |

\* LUDB is mostly used for 12-lead onset/offset detection; this work uses a single lead. Per-record outputs are in [`results/`](results).

## Features

- Preprocessing with adaptive filtering and median baseline removal  
- Template generation using Pearson correlation  
- Phasor transformation for peak enhancement  
- Detection of dissociated P-waves (P_dis)  
- Evaluation with MIT-BIH, BUTDB, QTDB AND LUDB using WFDB annotations  
- Visualization tools for template and peak analysis

## Installation

The package is not published on PyPI. Install it locally from a clone of this repository:

```bash
git clone https://github.com/NethmalWDI/ECG_P_wave_detection.git
cd ECG_P_wave_detection
pip install -e .
```

## Usage

Here's a basic example to detect P-waves and analyse for databases:

```python

from p_detection import ludb, but, bih, qtdb
import os


folder_path = 'path/to/BUT_PDB'
folder_path1 = 'path/to/LUDB'
folder_path2 = 'path/to/MIT_BIH'
folder_path3 = 'path/to/QTDB'


p = 0.8

ludb(folder_path=folder_path1, p_co=p)
bih(folder_path=folder_path2, p_co=p)
qtdb(folder_path=folder_path3, p_co=p)
but(folder_path=folder_path, p_co=p)
```

To run detection on your own records, use `detect_p()` directly:

Here's a basic example to detect P-waves and analyse for databases:

```python


from p_detection import detect_p
import os


folder_path = 'path to data'
record= ['name of the records']


p = 0.8

det_p,data=detect_p(folder_path=folder_path,record_name=record)

```

Example: running it in Google Colab
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
