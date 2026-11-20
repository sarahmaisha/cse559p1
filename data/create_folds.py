
from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold

RESPONSE_DIR = Path("curated_data/responses")
OUTPUT_FILE = Path("curated_data/data/fold_assignments.csv")

# Fixed seed 
SEED = 42

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

all_assignments = []

for file in sorted(RESPONSE_DIR.glob("*.csv")):
    response = pd.read_csv(file)

    response = response.sort_values("ModelID").reset_index(drop=True)

    drug_id = response["DRUG_ID"].iloc[0]

    assert response["ModelID"].is_unique

    response["fold"] = -1

    for fold, (train_idx, test_idx) in enumerate(kf.split(response)):
        response.loc[test_idx, "fold"] = fold

    assert (response["fold"] >= 0).all()

    all_assignments.append(
        response[["DRUG_ID", "ModelID", "fold"]]
    )

folds = pd.concat(all_assignments, ignore_index=True)

assert not folds.duplicated(["DRUG_ID", "ModelID"]).any()

folds.to_csv(OUTPUT_FILE, index=False)

print("Saved:", OUTPUT_FILE)
print("Total assignments:", len(folds))
print("Number of drugs:", folds["DRUG_ID"].nunique())
print(folds.head(10))
