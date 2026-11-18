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

    # turn into tensors for pyTorch Geometric
    edge_index = torch.tensor([from_list, to_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float)

    return edge_index, edge_weight

def summarize_graph(edge_index, edge_weight, genes, idx_dict, targets):
    print("Number of genes:", len(genes))
    print("Number of edges:", edge_index.shape[1] // 2)
    print("Edge weight stats: min", edge_weight.min().item(), "max", edge_weight.max().item(), "mean", edge_weight.mean().item())
    degree = torch.bincount(edge_index[0], minlength=len(genes))
    print("Average degree:", degree.float().mean().item())
    print("Isolated genes:", (degree == 0).sum().item())

    for name in targets:
        position = idx_dict[name]
        deg = degree[position].item()
        print(f"{name}: degree {deg}")


if __name__ == "__main__":
    def section(title):
        print("\n" + "-" * 50)
        print(title)
        print("-" * 50)

    # ---- 1. gene order ----
    section("1. Gene order (load_gene_order)")
    genes, idx = load_gene_order("data/fake/CellLinesLandmarkGenes.csv")
    print("Number of genes:   ", len(genes))
    print("First 3 genes:     ", genes[:3])
    print("Position of G5:    ", idx["G5 (1005)"])
    print("Position of G60:   ", idx["G60 (1060)"])

    # ---- 2. edges table ----
    section("2. Edges table (load_edges)")
    edges = load_edges("data/fake/curated_interaction_edges.json", idx)
    print("Shape (rows, cols):", edges.shape)
    print("\nFirst 5 rows:")
    print(edges.head())
    print("\nColumn types:")
    print(edges.dtypes)

    # ---- 3. edge_index tensors ----
    section("3. Edge tensors (build_edge_index)")
    edge_index, edge_weight = build_edge_index(edges)
    print("edge_index shape:  ", edge_index.shape)
    print("edge_weight shape: ", edge_weight.shape)
    print("\nFirst 3 edges (original direction):")
    print(edge_index[:, :3])
    print("\nSame 3 edges (reversed direction, should be the rows above flipped):")
    print(edge_index[:, 800:803])

    # ---- 4. graph summary ----
    section("4. Graph summary (summarize_graph)")
    summarize_graph(edge_index, edge_weight, genes, idx, ["G5 (1005)", "G42 (1042)", "G100 (1100)"])

