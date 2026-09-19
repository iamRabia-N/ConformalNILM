# ConformalNILM

Code and records for the paper *State-Conditional Conformal Prediction for Nonintrusive Load Monitoring in Unseen Households* accepted at IEEE SUSTAIN 2026.

## Data

REFIT CLEAN electrical load measurements (Murray et al., 2017).
Download from https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements. 
The script reads the data from the DATA_DIR constant near the top. Point DATA_DIR at the folder holding the CLEAN_House*.csv files.

## Run

Code is tested on Kaggle with a T4 GPU. The full run takes about 4 hours. Training uses houses 2 and 3, validation uses house 9, test uses houses 5, 6, and 20. Seeds are fixed at 1, 2, 3. The script writes all output files to its working directory. In this repository, the same files are organized into results/ and predictions/. Requirements are torch, numpy, pandas, scipy, and matplotlib.

## Files

| Path | Description |
|:---|:---|
| results/results_main.csv | Tables I, II, and III of the paper |
| results/results_multi_house.csv | Table IV |
| results/threshold_sensitivity.csv | ON-threshold sensitivity, Table V, Section V-E |
| results/ablation_calsize.csv | Calibration size ablation, Section V-F |
| results/online_recal.csv | Online recalibration results, Section V-F |
| results/coverage_levels.csv | Coverage at nominal levels 0.80, 0.90, 0.95, supplementary, not tabulated in the paper |
| results/metadata.json | Normalization constants, channel mappings, run configuration |
| predictions/preds_*.npz | Per-window prediction records, both architectures, all five appliances |
| predictions/roll_wm_*.npy | Rolling coverage series behind Fig. 2 |

`results_multi_house.csv` also contains rows for house 13. House 13 was evaluated with automatically detected channel assignments that could not be verified against the REFIT documentation. Houses 5 and 6 use documented assignments, so house 13 is not reported in the paper.

 GPU training is not bit-identical across runs, so regenerated numbers can differ slightly from the published tables. The published tables come from the prediction records in this repository.