import pandas as pd
from pathlib import Path
import numpy as np
import torch

# load expression (indexed by cell line), responses, and folds
def load_tables(folder):
    folder = Path(folder)
    expr = pd.read_csv(folder / "CellLinesLandmarkGenes.csv", index_col=0)
    responses = pd.read_csv(folder / "responses.csv")
    folds = pd.read_csv((folder / "folds.csv"))
    return expr, responses, folds

# Return one row per tested cell line for this drug: cell_line, ln_ic50, fold
def get_drug_rows(drug, responses, folds):
    r = responses[responses["drug"] == drug]
    f = folds[folds["drug"] == drug]

    # merge them on drug + cell_line
    df = r.merge(f, on=["drug", "cell_line"], validate="one_to_one")

    # raise error if missing fold or response
    if len(df) != len(r) or len(df) != len(f):
        raise ValueError(
            f"{drug}: {len(r)} responses, {len(f)} fold entries, "
            f"but only {len(df)} matched after merging"
        )
    return df


# Return X/y arrays for train, validation, and test for one fold.
def split_fold(df, expr, fold, val_frac=0.2, seed=0):
    # row i = df's cell line i
    X = expr.loc[df["cell_line"]].to_numpy()
    # item i = df's cell line i
    y = df["ln_ic50"].to_numpy()

    # test = this fold; rest = the other 4 folds
    is_test = (df["fold"] == fold).to_numpy()
    X_test, y_test = X[is_test], y[is_test]
    X_rest, y_rest = X[~is_test], y[~is_test]

    # fraction of the rest (val_frac) to be used for validation
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X_rest))
    n_val = int(len(X_rest) * val_frac)
    val_idx = order[:n_val]
    train_idx = order[n_val:]

    X_train, y_train = X_rest[train_idx], y_rest[train_idx]
    X_val, y_val = X_rest[val_idx], y_rest[val_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test

# return mu, sd for saving
def standardize(X_train, X_val, X_test):
    # use training mean/std to prevent leaking
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)

    # guard against dividing by zero for genes that never vary
    sd[sd == 0] = 1

    # standarize gene values
    X_train = (X_train - mu) / sd
    X_val   = (X_val   - mu) / sd
    X_test  = (X_test  - mu) / sd

    return X_train, X_val, X_test, mu, sd

# Convert each NumPy array to a float32 torch tensor
def to_tensors(*arrays):
    return [torch.tensor(a, dtype=torch.float32) for a in arrays]

# combine all helper functions to prepare fold
def prepare_fold(drug, fold, expr, responses, folds, val_frac=0.2, seed=0):
    df = get_drug_rows(drug, responses, folds)
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_fold(df, expr, fold, val_frac, seed)
    X_tr, X_val, X_te, mu, sd = standardize(X_tr, X_val, X_te)
    tensors = to_tensors(X_tr, y_tr, X_val, y_val, X_te, y_te)
    return tensors, mu, sd

if __name__ == "__main__":
    expr, responses, folds = load_tables("data/fake")
    print(expr.shape)          # expect (400, 200)
    print(expr.index[:3])      # first 3 cell line IDs
    print(responses.head())
    print(folds.head())

    df = get_drug_rows("drugA", responses, folds)
    print(df.head())
    print(len(df))                          # expect 359 - drugA's count from fake_data.py
    print(df["fold"].value_counts())        # how many cell lines per fold expect ~72

    X_tr, y_tr, X_val, y_val, X_te, y_te = split_fold(df, expr, fold=0)
    print("train:", X_tr.shape, y_tr.shape)    # expect (230, 200) (230,)
    print("val:  ", X_val.shape, y_val.shape)  # expect (57, 200)  (57,)
    print("test: ", X_te.shape, y_te.shape)    # expect (72, 200)  (72,)

    X_tr, X_val, X_te, mu, sd = standardize(X_tr, X_val, X_te)
    print("train mean (first 3 genes):", X_tr.mean(axis=0)[:3])    # ≈ 0
    print("train std  (first 3 genes):", X_tr.std(axis=0)[:3])     # ≈ 1
    print("test mean  (first 3 genes):", X_te.mean(axis=0)[:3])    # close to 0, but not exactly to verify that there is no leakage

    X_tr, y_tr, X_val, y_val, X_te, y_te = to_tensors(X_tr, y_tr, X_val, y_val, X_te, y_te)
    print(type(X_tr), X_tr.dtype, X_tr.shape)    # torch.Tensor, torch.float32, [230, 200]

