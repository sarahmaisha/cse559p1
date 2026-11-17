"""
Takes gene expression data and builds a graph for GNN training using GCNConv

load CellLinesLandmarkGenes.csv and take gene order
load curated_interaction_edges.json and turn gene names into positions add to graph
build edge_index and edge_weight arrays with both directions
"""

import json
import torch
import pandas as pd

# GCNConv relies on order of expression table, load gene order and index dictionary
def load_gene_order(path):
    genes = pd.read_csv(path, nrows=0).columns.to_list()[1:]
    idx_dict = {gene: i for i, gene in enumerate(genes)}
    return genes, idx_dict

# Create edge list from curated_interaction_edges.json, using index dictionary to convert gene names to positions
def load_edges(path, idx_dict):
  with open(path) as f:
    edges = json.load(f)
  edges = pd.DataFrame(edges, columns=["gene_a", "gene_b", "weight"])

  # source (gene_a) and target (gene_b), and weight columns
  edges["a"] = [idx_dict[g] for g in edges["gene_a"]]
  edges["b"] = [idx_dict[g] for g in edges["gene_b"]]
  edges["weight"] = edges["weight"].astype(float)

  return edges

# Build edge index and edge weight as torch tensors, with both directions for each edge
def build_edge_index(edges):
    a = edges["a"].to_list()
    b = edges["b"].to_list()

    from_list = a + b
    to_list = b + a

    weights = edges["weight"].to_list()
    weight_list = weights + weights

    # turn into tensors for pytorch geometric
    edge_index = torch.tensor([from_list, to_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float)

    return edge_index, edge_weight