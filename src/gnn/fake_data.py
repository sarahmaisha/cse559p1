import json
from pathlib import Path

import numpy as np
import pandas as pd

# set the random seed for reproducibility
rng = np.random.default_rng(42)

# create gene names based on column names
# ["G0 (1000), G1 (1001), ..., G199 (1199)"]
genes = [f"G{i} ({1000+i})" for i in range(200)]

# 1. make expression values

# create cell line ids based on DepMap IDs
# ["ACH-000000", "ACH-000001", ..., "ACH-000399"]
cell_lines = [f"ACH-{j:06d}" for j in range(400)]
gene_means = rng.uniform(1, 8, size=200)
# randomly generate expression values for each cell line and gene
values = gene_means + rng.normal(0, 1.5, size=(400, 200))
values = np.clip(values, 0, None)

# turn into table
expressions = pd.DataFrame(values, columns=genes)
expressions.insert(0, "ModelID", cell_lines)

# export to csv (follows daniel code)
out = Path("data/fake")
out.mkdir(parents=True, exist_ok=True)

expressions.to_csv(out / "CellLinesLandmarkGenes.csv", index=False)

# 2. create edges - random gene pairs with random weights
edges = []
# prevent duplicate or redundant pairs
seen = set()
while len(edges) < 800:
    i, j = rng.choice(200, size=2, replace=False)
    pair = (min(i, j), max(i, j))
    if pair in seen:
        continue
    seen.add(pair)
    weight = round(float(rng.uniform(0, 1)), 3)
    edges.append([genes[i], genes[j], weight])


# save as json
with open(out / "curated_interaction_edges.json", "w") as f:
    json.dump(edges, f)


# 3. make drug response values

# standarize gene values first
z = (values - values.mean(axis=0)) / values.std(axis=0)

# create dummy target gene - G5
nbrs_A = [b for (a, b) in seen if a == 5] + [a for (a, b) in seen if b == 5]

# plant dummy drug response signals for G5 and neighbors
n = len(cell_lines)
y_A = 3 - 1.0 * z[:, 5]  - 0.5 * z[:, nbrs_A].mean(axis=1) + rng.normal(0, 1, n)
y_B = 3 - 1.0 * z[:, 42] + rng.normal(0, 1, n)
y_C = 3 + rng.normal(0, 1, n)

# randomly drop 10% of the responses for each drug to simulate missing data
parts = []
for drug, y in {"drugA": y_A, "drugB": y_B, "drugC": y_C}.items():
    tested = rng.random(n) > 0.10
    parts.append(pd.DataFrame({
        "drug": drug,
        "cell_line": np.array(cell_lines)[tested],
        "ln_ic50": np.round(y[tested], 3),
    }))
responses = pd.concat(parts, ignore_index=True)
responses.to_csv(out / "responses.csv", index=False)

# drug list
drug_list = pd.DataFrame({
    "drug": ["drugA", "drugB", "drugC"],
    "targets": ["G5", "G42", "G100"],
    "control_label": ["positive", "positive", "negative"],
})
# export to csv
drug_list.to_csv(out / "drug_list.csv", index=False)

# debugging prints
print("G5 neighbors:", [genes[k] for k in nbrs_A])
print(responses.groupby("drug").size())
print("G5 vs drugA:  ", np.corrcoef(values[:, 5], y_A)[0, 1])
print("G42 vs drugB: ", np.corrcoef(values[:, 42], y_B)[0, 1])
print("G100 vs drugC:", np.corrcoef(values[:, 100], y_C)[0, 1])


# 4. create folds - for each drug, split its tested cell lines into 5 equal groups
fold_parts = []
for drug, group in responses.groupby("drug"):
    shuffled = rng.permutation(group["cell_line"].to_numpy())
    fold_ids = np.arange(len(shuffled)) % 5
    fold_parts.append(pd.DataFrame({
        "drug": drug,
        "cell_line": shuffled,
        "fold": fold_ids,
    }))
folds = pd.concat(fold_parts, ignore_index=True)
folds.to_csv(out / "folds.csv", index=False)

print(folds.groupby(["drug", "fold"]).size().unstack())
print(len(folds) == len(responses))


