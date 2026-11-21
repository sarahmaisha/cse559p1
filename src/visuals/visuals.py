import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd
from pathlib import Path

def plotPredVsTrue(y_true, y_pred, drug_name):

    s_coeff = spearmanr(y_true, y_pred).statistic
    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.75, color='steelblue', edgecolors='k',label=f'Spearman ρ = {s_coeff:.2f}')
    ax.legend()
    ax.set_title(f"True v.s. Predicted LN_IC50 values for {drug_name}")
    ax.set_xlabel("True Value")
    ax.set_ylabel("Predicted Value")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

predictions = pd.read_csv(Path("results/predictions/gnn_real_v3_teamfolds.csv"))
Venetoclax_preds = predictions[predictions['drug'] == 1909]
y_true = Venetoclax_preds['y_true']
y_pred = Venetoclax_preds['y_pred']
print(y_true)
plotPredVsTrue(y_true, y_pred, "Venetoclax")