"""Train the GNN for every drug x fold and save out-of-fold test predictions."""
from pathlib import Path
import pandas as pd
import torch

from graph import load_gene_order, load_edges, build_edge_index
from data import load_tables, get_drug_rows, prepare_fold
from model import GeneGCN
from train import fit, predict
from metrics import all_metrics

DATA = Path("data/fake")  # need to change this later!!
OUT = Path("results/predictions")
SEED = 0

def main():
    # load everything once
    genes, idx = load_gene_order(DATA / "CellLinesLandmarkGenes.csv")
    edge_index, edge_weight = build_edge_index(load_edges(DATA / "curated_interaction_edges.json", idx))
    expr, responses, folds = load_tables(DATA)
    drugs = pd.read_csv(DATA / "drug_list.csv")["drug"].to_list()

    parts = []
    for drug in drugs:
        df = get_drug_rows(drug, responses, folds)
        for fold in sorted(df["fold"].unique()):
            torch.manual_seed(SEED)
            # prepare this drug + fold
            (X_tr, y_tr, X_val, y_val, X_te, y_te), mu, sd = prepare_fold(drug, fold, expr, responses, folds)
            model = GeneGCN(edge_index, edge_weight, n_genes=len(genes), skip=True)
            model, _ = fit(model, X_tr, y_tr, X_val, y_val)
            y_pred = predict(model, X_te)

            test_lines = df.loc[df["fold"] == fold, "cell_line"].to_numpy()
            parts.append(pd.DataFrame({
                "drug": drug,
                "cell_line": test_lines,
                "fold": fold,
                "y_true": y_te.numpy(),
                "y_pred": y_pred,
            }))
            m = all_metrics(y_te.numpy(), y_pred)
            print(f"{drug} fold {fold}: n={len(y_te)}  rmse {m['rmse']:.3f}  r2 {m['r2']:.3f}  spearman {m['spearman']:.3f}")

    preds = pd.concat(parts, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    preds.to_csv(OUT / "gnn.csv", index=False)
    print(f"saved {len(preds)} predictions to {OUT / 'gnn.csv'}")

if __name__ == "__main__":
    main()
