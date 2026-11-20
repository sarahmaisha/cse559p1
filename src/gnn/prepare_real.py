"""Convert Daniel's curated data into the GNN pipeline's format:
responses.csv (drug-name-merged, replicates averaged), folds.csv (per cell line), drug_list.csv (controls)."""
from pathlib import Path
import numpy as np
import pandas as pd

FOLDER = Path("data/real_v2/curated_data")
SEED = 559
N_FOLDS = 5

# Preliminary control subset (proposal classes), all with targets already in the current gene panel.
# To be replaced by the rule-based drug list once the panel is rebuilt with all targets.
CONTROLS = pd.DataFrame([
    ("Venetoclax",  "BCL2",         "positive"),
    ("Olaparib",    "PARP1, PARP2", "positive"),
    ("Lapatinib",   "EGFR, ERBB2",  "positive"),
    ("Ruxolitinib", "JAK1, JAK2",   "positive"),
    ("Palbociclib", "CDK4, CDK6",   "negative"),
    ("Entinostat",  "HDAC1, HDAC3", "negative"),
], columns=["drug", "targets", "control_label"])

def make_responses(raw):
    """Merge drug IDs by name and average replicate screens -> drug, cell_line, ln_ic50."""
    merged = raw.groupby(["DRUG_NAME", "ModelID"])["LN_IC50"].mean().reset_index()
    merged = merged.rename(columns={"DRUG_NAME":"drug", "ModelID":"cell_line", "LN_IC50":"ln_ic50"})
    return merged

def make_folds(responses, seed=SEED, n_folds=N_FOLDS):
    """Assign each cell line one fold (same for every drug) -> drug, cell_line, fold."""
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
    folds = make_folds(responses)

    # sanity checks
    print("rows before merge:", len(raw), "| after merge:", len(responses))
    print("drugs:", responses["drug"].nunique(), "| cell lines:", responses["cell_line"].nunique())
    print("cell lines missing from expression:", (~responses["cell_line"].isin(expr_ids)).sum())
    print("control drugs found:", CONTROLS["drug"].isin(responses["drug"]).sum(), "of", len(CONTROLS))
    print(folds.drop_duplicates("cell_line")["fold"].value_counts().sort_index().to_dict())

    responses.to_csv(FOLDER / "responses.csv", index=False)
    folds.to_csv(FOLDER / "folds.csv", index=False)
    CONTROLS.to_csv(FOLDER / "drug_list.csv", index=False)
    print("wrote responses.csv, folds.csv, drug_list.csv to", FOLDER)

if __name__ == "__main__":
    main()
