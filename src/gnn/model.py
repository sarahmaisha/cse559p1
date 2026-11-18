import torch
import torch.nn as nn

class GeneGCN(nn.Module):
  def __init__(self, edge_index, edge_weight, n_genes, emb=8):
    super().__init__()
    # store edge_index and edge_weight as buffers so that they stay fixed
    # bc values stay same for every cell line
    self.register_buffer("edge_index", edge_index)
    self.register_buffer("edge_weight", edge_weight)

    # create gene embeddings
    # a Parameter of shape [n_genes, emb],
    # small random values so that gene values can be passed in
    self.gene_emb = nn.Parameter(torch.randn(n_genes, emb) * 0.1)

  # build list of features for every gene
  #   parameter - x: [batch, n_genes]
  #               batch is the group of cell lines
  def forward(self, x):
    # unsqueeze x to [batch, n_genes, 1]
    # wrap each value in its own list
    h = x.unsqueeze(-1)

    # expand the embeddings to [batch, n_genes, emb]
    # lines up tags (embeddings) with gene values
    e = self.gene_emb.expand(x.size(0), -1, -1)

    # concatenate -> [batch, n_genes, 1 + emb]
    return torch.cat([h, e], dim = -1)



if __name__ == "__main__":
  from graph import load_gene_order, load_edges, build_edge_index
  genes, idx = load_gene_order("data/fake/CellLinesLandmarkGenes.csv")
  edges = load_edges("data/fake/curated_interaction_edges.json", idx)
  edge_index, edge_weight = build_edge_index(edges)

  model = GeneGCN(edge_index, edge_weight, n_genes=len(genes))
  x = torch.randn(4, len(genes))
  out = model(x)
  print(out.shape)
