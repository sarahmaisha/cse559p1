import numpy as np
import pandas as pd

# the cell_ids need to be in the same order as X and Y
def load_folds(folds_path, drug_id, cell_ids):
    folds = pd.read_csv(folds_path)

    drug_folds = folds.loc[folds["DRUG_ID"] == drug_id, ["ModelID", "fold"]]

    # looks up each cell line's fold without changing order
    assignments = drug_folds.set_index("ModelID")["fold"]
    fold_ids = assignments.reindex(cell_ids)
    fold_ids = fold_ids.to_numpy(dtype=int)

    return fold_ids

# splits data into training and test sets for one fold
def get_fold_data(X, y, fold_ids, fold):
    #use selected fold for testing and rest for training
    train_mask = fold_ids != fold
    test_mask = fold_ids == fold

    # gene exp row for each
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test, test_mask
