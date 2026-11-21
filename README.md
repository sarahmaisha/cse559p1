# cse559p1
Project 1 for CSE 559 Team 6 — drug response prediction

## Setup
Requires **Python 3.12**. From the repo root:
```bash
uv sync
```
All commands below are run from the **repo root**.

## GNN pipeline (`src/gnn/`)

### Quick test on fake data
```bash
uv run python src/gnn/fake_data.py          # writes data/fake/
```
Then in `src/gnn/run_gnn.py` set `DATA = Path("data/fake")`, `FOLDS_FILE = "folds.csv"`, `EDGE_SCALE = 1.0`, and run steps 2–3 below.

### Real data: required files
`data/` is gitignored, so get these from the shared Google Drive folder:

```
data/real_v2/curated_data/
data/real_v2/fold_assignments.csv
```
Extract with: `mkdir -p data/real_v2 && tar -xf curated_data_v2.tar -C data/real_v2`

### Steps
```bash
# 1. convert data: writes responses.csv, fold_assignments_team.csv, drug_list.csv into data/real_v2/curated_data/
uv run python src/gnn/prepare_real.py

# 2. train every drug in drug_list.csv x 5 folds (~8 s per fold on a laptop; 6 drugs ≈ 5 min)
uv run python src/gnn/run_gnn.py

# 3. summary tables for that run
uv run python src/gnn/summarize.py results/predictions/<RUN_NAME>.csv
```

### Settings (top of `src/gnn/run_gnn.py`)
| Setting | Meaning |
|---|---|
| `DATA` | folder with the 5 input files |
| `FOLDS_FILE` | fold file inside `DATA` (`fold_assignments_team.csv` for real data) |
| `RUN_NAME` | names all outputs. **Change it for every new run**, or the previous one is overwritten |
| `EDGE_SCALE` | divides STRING scores (1000 for real data, 1 for fake) |
| `MODEL_KW`, `FIT_KW` | model and training hyperparameters (saved with each checkpoint) |

### Outputs
```
results/predictions/<RUN_NAME>.csv       out-of-fold test predictions: drug, cell_line, fold, y_true, y_pred
results/metrics/<RUN_NAME>_by_drug.csv   mean ± sd across folds (Spearman, RMSE, R²)
results/metrics/<RUN_NAME>_pooled.csv    metrics on all out-of-fold predictions per drug
results/metrics/<RUN_NAME>_folds.csv     one row per drug × fold
results/models/<RUN_NAME>/<drug>_fold<k>.pt   trained models (weights, config, gene order, standardization mu/sd)
```

## Using `summarize.py` for other models
Save out-of-fold **test** predictions as a CSV with columns `drug, cell_line, fold, y_true, y_pred`
(`drug` = DRUG_ID, `cell_line` = ModelID, folds from `fold_assignments.csv`), then:
```bash
uv run python src/gnn/summarize.py results/predictions/mlp.csv     # → results/metrics/mlp_*.csv
```
Needs `src/gnn/summarize.py` and `src/gnn/metrics.py` from branch `gnn_implementation`.

## Known limitations :/
- Target genes with protein-style names in GDSC (e.g. BCL-XL → BCL2L1, MEK1 → MAP2K1) are not yet in the gene panel; alias mapping pending.
- `drug_list.csv` currently holds 6 preliminary control drugs; the rule-based ~170-drug list comes after the panel rebuild.


