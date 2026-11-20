"""Per-drug summary of any model's out-of-fold predictions (columns: drug, cell_line, fold, y_true, y_pred)."""
import sys
from pathlib import Path
import pandas as pd
from metrics import all_metrics

# One row per (drug, fold) with its metrics
def per_fold(preds):
    rows = []
    for (drug, fold), g in preds.groupby(["drug", "fold"]):
        m = all_metrics(g["y_true"], g["y_pred"])
        rows.append({"drug": drug, "fold": fold, "n": len(g), **m})
    return pd.DataFrame(rows)

# Mean and sd of each metric across folds, one row per drug.
def per_drug(fold_table):
    return fold_table.groupby("drug")[["spearman", "rmse", "r2"]].agg(["mean", "std"])

# Metrics computed on all of a drug's out-of-fold predictions at once
def pooled(preds):
    rows = []
    for drug, g in preds.groupby("drug"):
        rows.append({"drug": drug, "n": len(g), **all_metrics(g["y_true"], g["y_pred"])})
    return pd.DataFrame(rows)

def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/predictions/gnn.csv")
    preds = pd.read_csv(path)

    folds_tbl = per_fold(preds)
    print(f"=== {path.name}: mean ± sd across folds ===")
    print(per_drug(folds_tbl).round(3))
    print(f"\n=== {path.name}: pooled out-of-fold ===")
    print(pooled(preds).round(3).to_string(index=False))

    out = Path("results/metrics")
    out.mkdir(parents=True, exist_ok=True)
    folds_tbl.to_csv(out / f"{path.stem}_folds.csv", index=False)
    print(f"\nsaved per-fold table to {out / (path.stem + '_folds.csv')}")

if __name__ == "__main__":
    main()
