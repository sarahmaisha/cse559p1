import torch
import torch.nn.functional as F
import copy

# One pass over the training data in shuffled mini-batches, returns the average loss
def train_one_epoch(model, optimizer, X, y, batch_size=32):

    model.train()
    n = len(X)

    # shuffle order
    order = torch.randperm(n)
    total_loss = 0.0
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        Xb, yb = X[idx], y[idx]

        # training steps
        pred = model(Xb).squeeze(-1)            # predict
        loss = F.mse_loss(pred, yb)             # measure error
        optimizer.zero_grad()                   # clear old gradients
        loss.backward()                         # gradient calculations
        optimizer.step()                        # apply gradients,move the weights

        # add up loss weighted by batch size
        total_loss += loss.item() * len(idx)
    return total_loss / n

# Average MSE on (X, y) with no learning
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).squeeze(-1)
        loss = F.mse_loss(pred, y)
    return loss.item()

# train with early stopping on validation loss
def fit(model, X_tr, y_tr, X_val, y_val, lr=1e-3, weight_decay=1e-4,
        max_epochs=200, patience=10, batch_size=32, verbose=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    history = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, optimizer, X_tr, y_tr, batch_size)
        
        # Early stopping for inner_cv training, which uses a validation set
        if X_val != None and y_val != None:
            val_loss = evaluate(model, X_val, y_val)
            history.append({"epoch": epoch, "best_epoch": best_epoch, "train_loss": train_loss, "val_loss": val_loss})
            if verbose:
                print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")

            if val_loss < best_val:
                # new best: update best_val, best_epoch, and snapshot the weights
                best_val = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

            elif epoch - best_epoch >= patience:
                # no improvement
                break
            
    if best_state != None:
        model.load_state_dict(best_state)
        print(f"best epoch {best_epoch}, val {best_val:.4f}")
        
    return model, history

# returns predictions
def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).squeeze(-1).numpy()

if __name__ == "__main__":
    from graph import load_gene_order, load_edges, build_edge_index
    from data import load_tables, prepare_fold
    from model import GeneGCN

    torch.manual_seed(0)
    genes, idx = load_gene_order("data/fake/CellLinesLandmarkGenes.csv")
    edge_index, edge_weight = build_edge_index(load_edges("data/fake/curated_interaction_edges.json", idx))
    expr, responses, folds = load_tables("data/fake")
    (X_tr, y_tr, X_val, y_val, X_te, y_te), mu, sd = prepare_fold("drugA", 0, expr, responses, folds)

    model = GeneGCN(edge_index, edge_weight, n_genes=len(genes))
    model, history = fit(model, X_tr, y_tr, X_val, y_val, verbose=True)
    print("test loss (best weights):", evaluate(model, X_te, y_te))


