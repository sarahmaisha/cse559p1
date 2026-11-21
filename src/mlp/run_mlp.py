import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from mlp import MLP
from train_mlp import fit, predict, evaluate, set_seed
from normalization import normalize_data
from cross_validation import load_folds, get_fold_data

ROOT = Path(__file__).resolve().parents[1]

SEED = 42
MAX_EPOCHS = 200
PATIENCE = 20

# load gene expression and drug response for one drug
def load_data(drug_id, data_dir):

    expression = pd.read_csv(data_dir / "CellLinesLandmarkGenes.csv", index_col=0)

    response = pd.read_csv(data_dir / "responses" / f"{drug_id}.csv")

    # matches the expression and response rows using ModelID
    X = expression.loc[response["ModelID"]].to_numpy()
    y = response["LN_IC50"].to_numpy()

    cell_ids = response["ModelID"].to_numpy()
    gene_names = expression.columns.tolist()

    return X, y, cell_ids, gene_names


def run_mlp(drug_id, data_dir, folds_file, results_dir):

    X, y, cell_ids, gene_names = load_data(drug_id, data_dir)

    fold_ids = load_folds(folds_file, drug_id, cell_ids)

    output_dir = results_dir / str(drug_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    histories = []

    # five outer folds
    for fold in range(5):

        print(f"Drug {drug_id}, Fold {fold + 1}/5")

        fold_seed = SEED + fold

        X_train, X_test, y_train, y_test, test_mask = (get_fold_data(X, y, fold_ids, fold))

        # splits the outer training and selects the best epoch
        X_inner, X_val, y_inner, y_val = train_test_split(X_train, y_train, test_size=0.20,random_state=fold_seed)

        # normalize data 
        X_inner_scaled, X_val_scaled, _ = normalize_data(X_inner, X_val)

        # Seed before initializing nn weights
        set_seed(fold_seed)
        validation_model = MLP(input_dim=X.shape[1])
        validation_model, history, best_epoch = fit(validation_model, X_inner_scaled, y_inner, X_val=X_val_scaled, y_val=y_val, epochs=MAX_EPOCHS, patience=PATIENCE, seed=fold_seed)
        print("Best epoch:", best_epoch)

        # normalized separately for the outer training set
        X_train_scaled, X_test_scaled, scaler = (normalize_data(X_train, X_test))
        set_seed(fold_seed)
        final_model = MLP(input_dim=X.shape[1])

        final_model, _, _ = fit(final_model, X_train_scaled, y_train, epochs=best_epoch, seed=fold_seed )

        # predict and evaluate 
        y_pred = predict(final_model, X_test_scaled)
 
        # save individual predictions
        for cell_id, actual, predicted in zip(cell_ids[test_mask], y_test, y_pred):
            predictions.append({
                "drug": drug_id,
                "cell_line": cell_id,
                "fold": fold,
                "y_true": float(actual),
                "y_pred": float(predicted)
            })

        # save training and validation loss history.
        for row in history:
            histories.append({
                "DRUG_ID": drug_id,
                "fold": fold,
                **row
            })

        # used for later SHAP analysis
        #torch.save(final_model.state_dict(), output_dir / f"model_fold_{fold}.pt")


    # save the out of fold predictions
    predictions_df = pd.DataFrame(predictions)

    
    assert len(predictions_df) == len(y)
    assert predictions_df["cell_line"].is_unique      # each cell line should be predicted once

    predictions_df.to_csv(output_dir / "predictions.csv", index=False )
    pd.DataFrame(histories).to_csv(output_dir / "training_history.csv", index=False)

    # get the exact gene name order for SHAP
    pd.DataFrame({"gene": gene_names}).to_csv(output_dir / "gene_names.csv", index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--drug-id", type=int, default=1003)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "curated_data")
    parser.add_argument("--folds", type=Path, default=ROOT / "data" / "fold_assignments.csv")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "mlp")

    args = parser.parse_args()

    run_mlp(args.drug_id, args.data_dir, args.folds, args.results_dir)
