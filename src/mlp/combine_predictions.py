# Combines the MLP predictions from the six selected drugs (controls) into one CSV.
# STRICTLY Used to generate summary tables for the INTERMEDIATE project results.

from pathlib import Path
import pandas as pd

drug_ids = [1909, 1017, 1558, 1507, 1054, 1593]

predictions = []

for drug_id in drug_ids:
    path = Path(f"results/mlp/{drug_id}/predictions.csv")

    if path.exists():
        df = pd.read_csv(path)
        predictions.append(df)
    else:
        print(f"Missing predictions for drug {drug_id}")

if not predictions:
    raise ValueError("No prediction files found.")

combined = pd.concat(predictions, ignore_index=True)

output = Path("results/predictions/mlp.csv")
output.parent.mkdir(parents=True, exist_ok=True)

combined.to_csv(output, index=False)

print(f"Saved {len(combined)} predictions to {output}")
