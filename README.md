# State-Conditional Conformal Prediction for Nonintrusive Load Monitoring in Unseen Households

Code and records for the paper submitted to IEEE SUSTAIN 2026.

## Data

REFIT CLEAN electrical load measurements (Murray et al., 2017).
Download from https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements
The dataset is not included in this repository. The script locates the data in two places, a DATA_DIR constant near the top and a second DATA_DIR set by a glob search in the multi-house section. Point both at the folder holding the CLEAN_House*.csv files.

## Run

Run sc_cqr_nilm.py top to bottom on a GPU machine. Tested on Kaggle with an NVIDIA T4. The full run takes about 4 hours. Training uses houses 2 and 3, validation house 9, test houses 20, 5, and 6. Seeds are fixed at 1, 2, 3. The script writes all output files to its working directory. In this repository the same files are organized into results/ and predictions/. Requirements are torch, numpy, pandas, scipy, and matplotlib.

## Files

| Path | Description |
|:---|:---|
| results/results_main.csv | Tables I, II, and III of the paper |
| results/results_multi_house.csv | Table IV |
| results/ablation_calsize.csv | Calibration size ablation, Section V-E |
| results/online_recal.csv | Online recalibration results, Section V-E |
| results/coverage_levels.csv | Coverage at nominal levels 0.80, 0.90, 0.95, supplementary, not tabulated in the paper |
| results/metadata.json | Normalization constants, channel mappings, run configuration |
| predictions/preds_*.npz | Per-window prediction records, both architectures, all five appliances |
| predictions/roll_wm_*.npy | Rolling coverage series behind Fig. 2 |

The files in predictions/ are derived from the REFIT CLEAN dataset (CC BY 4.0).

## Checkpoints

Trained model checkpoints exceed the file size limit here. A rerun of the script retrains them from the dataset with the fixed seeds. GPU training is not bit-identical across runs, so regenerated numbers can differ slightly from the published tables. The published tables come from the prediction records in this repository.
