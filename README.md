### Project 1 for CSE 449/559 Team 6

This project investigates whether deep learning models outperform simpler baselines in predicting drug response from gene expression data, and whether complex models identify biologically relevant drug targets better than direct measurement.

## Models

We compare four prediction models:
- Dummy Regressor: Predicts the mean ln(IC50) of the training data.
- Ridge Regression: Linear regression with L2 regularization.
- Multilayer Perceptron (MLP): Neural network trained on gene expression features.
- Graph Neural Network (GNN): Uses gene expression and protein-protein interaction data to predict drug response.

Models are evaluated using five-fold cross-validation with RMSE, R², and Spearman correlation. We also investigate gene importance and recovery of known drug targets.

## Setup
This project uses Python and [uv](https://docs.astral.sh/uv/).
```bash
uv sync
```
For setup in **SOL computing cluster**, see [mamba_env/README.md](mamba_env/README.md).

All commands below are run from the **repository root**

## Dataset
The preprocessed data is excluded from Git due to its size.

**Dataset download:** [(https://drive.google.com/drive/folders/1CMM9qp4JF7WliSxWvv1pLtNgHplTRtjL?usp=share_link)]

Navigate to 'running_experiments/curated_data.tar' in the Drive folder.
Each model's README describes the required dataset location and any additional preprocessing steps.

## Project Structure

```text
cse559p1/
├── src/
│   ├── mlp/               # Multilayer perceptron
│   └── gnn/               # Graph neural network
├── mamba_env/             # SOL environment setup
├── data/                  # Fold assignments and input data
├── results/               # Model predictions and metrics
├── basic_regressor.py     # Dummy and Ridge baselines
├── pyproject.toml         # Project dependencies
└── README.md
```

## Running the Models

Each model has its own implementation or detailed README instructions.

### Dummy and Ridge Regression (basic_regressor.py):

See the baseline implementation in the repository for training and evaluation. The baseline models use the same five-fold assignments as the deep learning models.

### MLP (src/mlp):

See [MLP README](src/mlp/README.md) for the training pipeline, normalization, cross-validation, and evaluation instructions.

### GNN (src/gnn):

See [GNN README](src/gnn/README.md) for graph preparation, model training, and evaluation instructions.


