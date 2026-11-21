# Multilayer Perceptron
## Setup
Requires **Python 3.14** and [uv](https://docs.astral.sh/uv/). From the repo root:
```bash
uv sync
```

Dependencies are installed from the repository's `pyproject.toml` and `uv.lock`.

## MLP pipeline (`src/mlp/`)

### Required files
- **Download curated_data.tar and fold_assignments.csv:** [https://drive.google.com/drive/folders/1iMITmJMri_N8v_ozLLIymE5zlmwjsLNl?usp=share_link]

- Exract curated_data.tar: `tar -xf curated_data.tar`

Place the following files under the repo root:

```
curated_data/CellLinesLandmarkGenes.csv
curated_data/responses/<DRUG_ID>.csv
data/fold_assignments.csv
```

The expression file contains cell-line gene expression profiles (rows indexed by `ModelID`, columns representing genes); each drug response file contains the matching `ModelID`, `DRUG_ID`, and `LN_IC50` values. The fold file assigns each `(DRUG_ID, ModelID)` pair to an cross validation fold (0–4).

### Reproducing the reported MLP results table

The following workflow reflext the **six drug, 5 fold MLP performance table** reported in the intermediate project report. Start with the source files and shared fold assignments listed above

```bash
# 1. Install dependencies (from the repo root)
uv sync

# 2. Train a separate MLP for each of the six control drugs
for drug in 1909 1017 1558 1507 1054 1593; do
    uv run python src/mlp/run_mlp.py --drug-id "$drug"
done

# 3. Combine only the out-of-fold test predictions
# The expected six drug IDs are 1909, 1017, 1558, 1507, 1054, and 1593. Check that all six per-drug predictions.csv files exist BEFORE combining them
uv run python src/mlp/combine_predictions.py

# 4. Recalculate the published evaluation metrics
uv run python src/mlp/summarize.py results/predictions/mlp.csv
```

Check these results after running:

- `results/predictions/mlp.csv` — one prediction per included cell line, with columns `drug, cell_line, fold, y_true, y_pred`.
- `results/metrics/mlp_by_drug.csv` — **mean and standard deviation over five folds** for RMSE, R², and Spearman correlation; this corresponds to the table reported in the intermediate paper.
- `results/metrics/mlp_pooled.csv` — metrics computed from all out-of-fold predictions pooled per drug; **these are different from the mean of the five fold metrics**.
- `results/metrics/mlp_folds.csv` — one metrics row per drug and outer fold.


### Individual run steps
```bash
# 1. Train one drug using five-fold cross-validation
uv run python src/mlp/run_mlp.py --drug-id 1909

# 2. Repeat for the six preliminary control drugs
uv run python src/mlp/run_mlp.py --drug-id 1017
uv run python src/mlp/run_mlp.py --drug-id 1558
uv run python src/mlp/run_mlp.py --drug-id 1507
uv run python src/mlp/run_mlp.py --drug-id 1054
uv run python src/mlp/run_mlp.py --drug-id 1593

# 3. Combine the six per-drug prediction files for intermediate results
uv run python src/mlp/combine_predictions.py

# 4. Generate the common evaluation tables
uv run python src/mlp/summarize.py results/predictions/mlp.csv
```

The six preliminary drugs are:

| DRUG_ID | Drug | Target genes | Proposed control |
|---|---|---|---|
| 1909 | Venetoclax | BCL2 | Positive |
| 1017 | Olaparib | PARP1, PARP2 | Positive |
| 1558 | Lapatinib | EGFR, ERBB2 | Positive |
| 1507 | Ruxolitinib | JAK1, JAK2 | Positive |
| 1054 | Palbociclib | CDK4, CDK6 | Negative |
| 1593 | Entinostat | HDAC1, HDAC3 | Negative |

### Settings (`src/mlp/run_mlp.py`)

| Setting | Meaning |
|---|---|
| `SEED` | Random seed (currently 42, plus fold index) |
| `MAX_EPOCHS` | Maximum epochs used for inner validation (currently 200) |
| `PATIENCE` | Early-stopping patience during inner validation (currently 20) |
| `--drug-id` | Drug to train (default 1003) |
| `--data-dir` | Directory containing expression data and `responses/` |
| `--folds` | CSV with shared fold assignments |
| `--results-dir` | Directory for per-drug MLP outputs (default `results/mlp/`) |

Each outer fold holds out one-fifth of the cell lines for testing. The remaining cells are split into inner training/validation data to select a training epoch. A new model is then trained on the entire outer training set for that many epochs and evaluated only on the held-out fold. Min-max scaling is fitted on training data only.

### Outputs

```
results/mlp/<DRUG_ID>/predictions.csv         out-of-fold predictions: drug, cell_line, fold, y_true, y_pred
results/mlp/<DRUG_ID>/training_history.csv    inner training and validation losses by epoch
results/mlp/<DRUG_ID>/gene_names.csv          gene names in input-feature order

results/predictions/mlp.csv                   combined predictions for the six control drugs
results/metrics/mlp_by_drug.csv               mean ± sd across folds (Spearman, RMSE, R²)
results/metrics/mlp_pooled.csv                metrics across all out-of-fold predictions per drug
results/metrics/mlp_folds.csv                 one row per drug × fold
```

### Code files

| File | Purpose |
|---|---|
| `src/mlp/mlp.py` | Defines the MLP architecture |
| `src/mlp/train_mlp.py` | Trains the MLP, performs early stopping, and predicts responses |
| `src/mlp/normalization.py` | Fits training-only min-max normalization and transforms validation/test data |
| `src/mlp/cross_validation.py` | Loads shared fold assignments and creates outer train/test splits |
| `src/mlp/run_mlp.py` | Runs one drug across five folds and saves predictions/checkpoints |
| `src/mlp/combine_predictions.py` | Combines predictions for the six intermediate-results drugs |
| `src/mlp/metrics.py` | Calculates RMSE, R², and Spearman correlation |
| `src/mlp/summarize.py` | Generates per-fold, per-drug, and pooled metrics tables |
