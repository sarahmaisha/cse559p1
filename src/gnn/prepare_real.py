"""Convert Daniel's curated data into the GNN pipeline's format:
responses.csv (drug-name-merged, replicates averaged), folds.csv (per cell line), drug_list.csv (controls)."""
from pathlib import Path
import numpy as np
import pandas as pd

FOLDER = Path("data/real_v2/curated_data")
SEED = 559
N_FOLDS = 5

# Preliminary control subset (proposal classes), all with targets already in the current gene panel.
CONTROLS = pd.DataFrame([
    (1909, "Venetoclax",  "BCL2",         "positive"),
    (1017, "Olaparib",    "PARP1, PARP2", "positive"),
    (1558, "Lapatinib",   "EGFR, ERBB2",  "positive"),
    (1507, "Ruxolitinib", "JAK1, JAK2",   "positive"),
    (1054, "Palbociclib", "CDK4, CDK6",   "negative"),
    (1593, "Entinostat",  "HDAC1, HDAC3", "negative"),
], columns=["drug", "name", "targets", "control_label"])

# One row per (DRUG_ID, cell line) -> drug, cell_line, ln_ic50
def make_responses(raw):
    return raw.rename(columns={"DRUG_ID": "drug", "ModelID": "cell_line", "LN_IC50": "ln_ic50"})[["drug", "cell_line", "ln_ic50"]]

# Assign each cell line one fold (same for every drug) -> drug, cell_line, fold
def make_folds(responses, seed=SEED, n_folds=N_FOLDS):
    rng = np.random.default_rng(seed)
    cell_lines = sorted(responses["cell_line"].unique())
    shuffled = rng.permutation(cell_lines)
    fold_of = dict(zip(shuffled, np.arange(len(shuffled)) % n_folds))
    folds = responses[["drug", "cell_line"]].copy()
    folds["fold"] = responses["cell_line"].map(fold_of)
    return folds

def main():
    raw = pd.read_csv(FOLDER / "ResponseCellLineMerge.csv")
    expr_ids = set(pd.read_csv(FOLDER / "CellLinesLandmarkGenes.csv", usecols=[0]).iloc[:, 0])

    responses = make_responses(raw)

    # folds = make_folds(responses)

    # replace ^^ with fold assignments
    folds = pd.read_csv(FOLDER.parent / "fold_assignments.csv").rename(columns={"DRUG_ID": "drug", "ModelID": "cell_line"})


    # sanity checks
    print("rows before merge:", len(raw), "| after merge:", len(responses))
    print("drugs:", responses["drug"].nunique(), "| cell lines:", responses["cell_line"].nunique())
    print("cell lines missing from expression:", (~responses["cell_line"].isin(expr_ids)).sum())
    print("control drugs found:", CONTROLS["drug"].isin(responses["drug"]).sum(), "of", len(CONTROLS))
    print(folds.drop_duplicates("cell_line")["fold"].value_counts().sort_index().to_dict())

    responses.to_csv(FOLDER / "responses.csv", index=False)
    folds.to_csv(FOLDER / "fold_assignments_team.csv", index=False)
    CONTROLS.to_csv(FOLDER / "drug_list.csv", index=False)
    print("wrote responses.csv, fold_assignments_team.csv, drug_list.csv to", FOLDER)

if __name__ == "__main__":
    main()
