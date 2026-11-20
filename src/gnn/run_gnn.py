"""Train the GNN for every drug x fold and save out-of-fold test predictions."""
from pathlib import Path
import pandas as pd
import torch
import time

from graph import load_gene_order, load_edges, build_edge_index
from data import load_tables, get_drug_rows, prepare_fold
from model import GeneGCN
from train import fit, predict
from metrics import all_metrics
from checkpoint import save_checkpoint

DATA = Path("data/real_v2/curated_data")
RUN_NAME = "gnn_real_v2"
MODEL_DIR = Path("results/models") / RUN_NAME
SEED = 0
EDGE_SCALE = 1000.0

# saved settings for the model
MODEL_KW = dict(emb=8, hidden=16, dropout=0.3, skip=True)
FIT_KW = dict(lr=1e-3, weight_decay=1e-4, max_epochs=200, patience=10)
OUT = Path("results/predictions")


def main():
    # load everything once
    genes, idx = load_gene_order(DATA / "CellLinesLandmarkGenes.csv")
    edge_index, edge_weight = build_edge_index(load_edges(DATA / "curated_interaction_edges.json", idx))
    edge_weight = edge_weight / EDGE_SCALE
    expr, responses, folds = load_tables(DATA)
    drugs = pd.read_csv(DATA / "drug_list.csv")["drug"].to_list()

    parts = []

    # temp pilot
    # drugs = pd.read_csv(DATA / "drug_list.csv")["drug"].to_list()[:1]

    for drug in drugs:
        df = get_drug_rows(drug, responses, folds)
        for fold in sorted(df["fold"].unique()):
            t0 = time.time()
            torch.manual_seed(SEED)
            # prepare this drug + fold
            (X_tr, y_tr, X_val, y_val, X_te, y_te), mu, sd = prepare_fold(drug, fold, expr, responses, folds)
            model = GeneGCN(edge_index, edge_weight, n_genes=len(genes), **MODEL_KW)
            model, _ = fit(model, X_tr, y_tr, X_val, y_val, **FIT_KW)
            y_pred = predict(model, X_te)

            # save model configuration
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            save_checkpoint(MODEL_DIR / f"{drug}_fold{fold}.pt", model, MODEL_KW,
                            genes, mu, sd, drug, fold, SEED)

            test_lines = df.loc[df["fold"] == fold, "cell_line"].to_numpy()
            parts.append(pd.DataFrame({
                "drug": drug,
                "cell_line": test_lines,
                "fold": fold,
                "y_true": y_te.numpy(),
                "y_pred": y_pred,
            }))
            m = all_metrics(y_te.numpy(), y_pred)
            print(f"{drug} fold {fold}: n={len(y_te)}  rmse {m['rmse']:.3f}  r2 {m['r2']:.3f}  spearman {m['spearman']:.3f} time {time.time() - t0:.0f}s")

    preds = pd.concat(parts, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    preds.to_csv(OUT / f"{RUN_NAME}.csv", index=False)
    print(f"saved {len(preds)} predictions to {OUT / f'{RUN_NAME}.csv'}")

if __name__ == "__main__":
    main()
