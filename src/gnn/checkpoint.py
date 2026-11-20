import torch
from model import GeneGCN

# Save configuration of model
def save_checkpoint(path, model, config, genes, mu, sd, drug, fold, seed):
    torch.save({
        "state_dict": model.state_dict(),
        "config": config,
        "genes": genes,
        "mu": torch.tensor(mu, dtype=torch.float32),
        "sd": torch.tensor(sd, dtype=torch.float32),
        "drug": drug,
        "fold": int(fold),
        "seed": seed,
    }, path)

# Rebuild the model from loaded configuration (checkpoint)
def load_checkpoint(path):
    ckpt = torch.load(path, weights_only=True)
    sd_ = ckpt["state_dict"]
    model = GeneGCN(sd_["edge_index"], sd_["edge_weight"],
                    n_genes=len(ckpt["genes"]), **ckpt["config"])
    model.load_state_dict(sd_)
    model.eval()
    return model, ckpt

if __name__ == "__main__":
    import numpy as np
    from data import load_tables, prepare_fold
    from train import predict
    expr, responses, folds = load_tables("data/fake")
    (_, _, _, _, X_te, _), mu, sd = prepare_fold("drugB", 0, expr, responses, folds)
    model, ckpt = load_checkpoint("results/models/gnn/drugB_fold0.pt")
    print("drug/fold:", ckpt["drug"], ckpt["fold"], " config:", ckpt["config"])
    print("mu matches:", np.allclose(ckpt["mu"].numpy(), mu))
    print("first 3 predictions:", predict(model, X_te)[:3])

