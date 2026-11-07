# DeepONet-HPA-Metabolic-UKB

This repository contains the code for: **"A Deep Operator Network Surrogate for Individualized HPA-Metabolic Simulation at UK Biobank Scale"** (extended ICCS 2026 paper, submitted to the Journal of Computational Science). It takes the ICCS pilot ([HPA-Metabolic-UKB](https://github.com/rezi-getsadze/HPA-Metabolic-UKB)) from 100 patients to N=1,798 UK Biobank participants (1,000 healthy, 500 IFG, 298 T2D) and trains a DeepONet surrogate on the coupled 18-variable HPA-metabolic ODE, so a stability grid that takes about two hours on the solver runs in seconds.

### Main files

* **`synthetic_data.py`**: generates UKB-shaped synthetic cohorts, Olink CSVs and a synthetic `sweep_results_full.csv`, so everything runs without biobank access.
* **`t2d_preprocess.py`**: the data pipeline. medication-naive filter (47 drug codes), stress score from the 16 MHQ items, glucose unit conversion, CRP NaN fill, Olink merge and prioritised sampling. writes the three cohorts to `Training Data/`.
* **`t2d_ode_uncalibrated_v2.py`**: the baseline with population-wide constants, where everyone collapses to one attractor.
* **`t2d_ode_calibrated_v2.py`**: individualized scaling of glucose uptake and hepatic production on the full cohort.
* **`t2d_deeponet_v8.py`**: the main script. trains the 5-day and 90-day surrogates, runs the cohort, HbA1c and longitudinal checks, the stability landscape and the ODE spot-check, and builds Figure 2 (`output/panel2.svg`).
* **`t2d_deeponet_multiseed.py`**: retrains the 5-day surrogate under five seeds for panels K-M.
* **`t2d_cohort_fractions_ci.py`**: Wilson 95% CIs for the H-J cohort fractions.
* **`t2d_deeponet_v7.py`**: the 120,000-run one-year sweep (local test, Snellius job files, analysis) and the racor stability sweep.
* **`t2d_deeponet_v10.py`**: the eleven-feature Olink variant.

### Run order

```bash
python synthetic_data.py
python t2d_preprocess.py
python t2d_ode_uncalibrated_v2.py
python t2d_ode_calibrated_v2.py
python t2d_deeponet_v8.py
python t2d_deeponet_multiseed.py
python t2d_cohort_fractions_ci.py
python t2d_deeponet_v8.py
python t2d_deeponet_v7.py
python t2d_deeponet_v10.py
```

v8 runs twice on purpose. multiseed and the CI script both read v8's cache, and the second v8 run picks up the multi-seed cache for panels K-M. caches go to `cache/`, figures to `output/` and `runtime/`, and most scripts also save what they print under `runtime/`.

### Data note

No data is included. `synthetic_data.py` writes randomized cohorts with the raw UK Biobank column structure, so all scripts run end to end, but the numbers will not match the paper. Raw UK Biobank data can't be shared due to licensing; access is by application at [ukbiobank.ac.uk](https://www.ukbiobank.ac.uk/). The real sweep output (`sweep_results_full.csv`, 120,000 rows) will be archived on Zenodo on publication.

### Installation

1. clone the repo:
   ```bash
   git clone https://github.com/rezi-getsadze/DeepONet-HPA-Metabolic-UKB.git
   ```
2. to install the required Python packages, run:
   ```bash
   pip install -r requirements.txt
   ```
