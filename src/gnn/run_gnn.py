"""Train the GNN for every drug x fold and save out-of-fold test predictions."""
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import time
import math
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


from graph import load_gene_order, load_edges, build_edge_index
from data import load_tables, get_drug_rows, prepare_fold, standardize
from model import GeneGCN
from train import fit, predict
from metrics import all_metrics
from checkpoint import save_checkpoint

DATA = Path("curated_data")
RUN_NAME = "gnn_real_v3_teamfolds"
MODEL_DIR = Path("results/models") / RUN_NAME
SEED = 0
EDGE_SCALE = 1000.0


# Hyperparameters to tune during nested cross-validation
param_grid = [
    ({"emb": 8,  "hidden": 8,  "dropout": 0.2}, {"lr": 1e-3, "weight_decay": 1e-4, "patience": 10}),
    # ({"emb": 8,  "hidden": 16, "dropout": 0.3}, {"lr": 1e-3, "weight_decay": 1e-4 "patience": 10}),
    # ({"emb": 12, "hidden": 16, "dropout": 0.4}, {"lr": 1e-3, "weight_decay": 1e-4, "patience": 10}),
    # ({"emb": 16, "hidden": 16, "dropout": 0.3}, {"lr": 1e-3, "weight_decay": 1e-4, "patience": 10}),
    # ({"emb": 16, "hidden": 32, "dropout": 0.4}, {"lr": 1e-3, "weight_decay": 1e-4, "patience": 10}),
    # ({"emb": 8,  "hidden": 16, "dropout": 0.3}, {"lr": 5e-4, "weight_decay": 1e-4, "patience": 10}),
    # ({"emb": 8,  "hidden": 16, "dropout": 0.3}, {"lr": 5e-3, "weight_decay": 1e-4, "patience": 10}),
    # ({"emb": 8,  "hidden": 16, "dropout": 0.3}, {"lr": 1e-3, "weight_decay": 1e-3, "patience": 10}),
    # ({"emb": 8,  "hidden": 16, "dropout": 0.3}, {"lr": 1e-3, "weight_decay": 1e-5, "patience": 10}),
]
            
OUT = Path("results/predictions")
FOLDS_FILE = "fold_assignments_team.csv"


def main():
    # load everything once
    genes, idx = load_gene_order(DATA / "CellLinesLandmarkGenes.csv")
    edge_index, edge_weight = build_edge_index(load_edges(DATA / "curated_interaction_edges.json", idx))
    edge_weight = edge_weight / EDGE_SCALE
    expr, responses, folds = load_tables(DATA, FOLDS_FILE)
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
            X_outer_tr, y_outer_tr, X_te, y_te = prepare_fold(drug, fold, expr, responses, folds)

            best_score = math.inf
            best_model_params = None
            best_fit_params = None
            best_epochs_for_params = None
            
            for model_params, fit_params in param_grid:
                # Prepare inner-folds
                print(f"Testing model config: {model_params} and learning config: {fit_params}")
                inner_results = []
                best_epochs = []
                inner_cv = KFold(n_splits=5,shuffle=True,random_state=SEED)
                
                for inner_train_idx, inner_val_idx in inner_cv.split(X_outer_tr):
                    X_tr = X_outer_tr[inner_train_idx]
                    X_val = X_outer_tr[inner_val_idx]
                    
                    y_tr = y_outer_tr[inner_train_idx]
                    y_val = y_outer_tr[inner_val_idx]
                    
                    # Need to standardize for the inner split
                    X_tr, X_val, mu, sd = standardize(X_tr, X_val)
                
                    model = GeneGCN(edge_index, edge_weight, n_genes=len(genes), **model_params)
                    model, history = fit(model, X_tr, y_tr, X_val, y_val, **fit_params)
                    best_epochs.append(history[-1]["best_epoch"])
                    
                    y_pred = predict(model, X_val)
                    score = mean_squared_error(y_val, y_pred)
                    inner_results.append(score)
                
                # Take the average score of these params across all 5 inner folds
                mean_score = np.mean(inner_results)
                
                # Score is loss, so a better score is lower
                if mean_score < best_score:
                    best_score = mean_score
                    best_model_params = model_params.copy()
                    best_fit_params = fit_params.copy()
                    best_epochs_for_params = best_epochs.copy()
            
            X_train, X_test, mu, sd = standardize(X_outer_tr, X_te)
            # Train a fresh model using all outer training samples with best params
            model = GeneGCN(edge_index, edge_weight, n_genes=len(genes), **best_model_params)
            # Use the median of the best epochs from best hyperparameters
            median_epochs = int(np.median(best_epochs_for_params))
            
            model, _ = fit(model, X_train, y_outer_tr, X_val=None, y_val=None, max_epochs=median_epochs,**best_fit_params)

            # Evaluate on outer test samples
            y_pred = predict(model, X_test)

            # save model configuration
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            save_checkpoint(MODEL_DIR / f"{drug}_fold{fold}.pt", model, best_model_params, 
                            best_fit_params, genes, mu, sd, drug, fold, SEED)

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
