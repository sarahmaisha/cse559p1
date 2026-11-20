import numpy as np
import torch
import copy
import random
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed=42):
    # setting some random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, optimizer, loss_fn):

    model.train()
    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()          # clear any gradients from the previous batch
        y_pred = model(X_batch)        # forward pass predicts LN_IC50
        loss = loss_fn(y_pred, y_batch) # Calculate mean squared error
        loss.backward()                 # complete backpropagation to compute gradients
        optimizer.step()            # update the trainable weights

        total_loss += loss.item() * len(X_batch)
        total_samples += len(X_batch)

    return total_loss / total_samples

# X should already be normalized
# returns a 1D array w one prediction per cell line
def predict(model, X):
    model.eval()
    X_tensor = torch.as_tensor(np.asarray(X, dtype=np.float32))
    with torch.no_grad():
        predictions = model(X_tensor)

    return predictions.numpy().reshape(-1)


def evaluate(model, X, y):
    y_pred = predict(model, X)

    rmse = np.sqrt(mean_squared_error(y, y_pred))   #lower RMSE is better
    r2 = r2_score(y, y_pred)                          #higher r2 is better
    spearman = spearmanr(y, y_pred).statistic   # higher Spearman cor is better

    return {
        "RMSE": float(rmse),
        "R2": float(r2),
        "Spearman": float(spearman)
    }


def fit(model, X_train, y_train, X_val=None, y_val=None, epochs=200, batch_size=32,learning_rate=0.001, weight_decay=0.01, patience=20, seed=42):

    set_seed(seed)

    X_tensor = torch.as_tensor(np.asarray(X_train, dtype=np.float32))
    y_tensor = torch.as_tensor(np.asarray(y_train, dtype=np.float32)).reshape(-1, 1)
    dataset = TensorDataset(X_tensor, y_tensor)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,generator=generator)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    has_validation = X_val is not None and y_val is not None

    best_loss = float("inf")
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, loader, optimizer, loss_fn)

        record = {
            "epoch": epoch,
            "train_mse": train_loss
        }

        # If there's validation data, then it stops early when it stops improving and restores the best model weights
        if has_validation:
            val_predictions = predict(model, X_val)
            val_loss = mean_squared_error(y_val, val_predictions)

            record["validation_mse"] = float(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch

                # save the weights at the best epoch.
                best_weights = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                history.append(record)
                break

        history.append(record)

    if has_validation:
        # get weights from the best validation epoch.
        model.load_state_dict(best_weights)
    else:
        best_epoch = epochs

    return model, history, best_epoch
