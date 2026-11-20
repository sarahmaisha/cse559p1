import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import root_mean_squared_error, r2_score

def all_metrics(y_true, y_pred):
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true = rng.normal(3, 1.5, 100)
    print("good model: ", all_metrics(y_true, y_true + rng.normal(0, 0.5, 100)))
    print("predict mean:", all_metrics(y_true, np.full(100, y_true.mean()) + rng.normal(0, 1e-9, 100)))

