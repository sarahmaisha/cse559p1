import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GeneGCN(nn.Module):
  def __init__(self, edge_index, edge_weight, n_genes, emb=8, hidden=16, dropout=0.3):
    super().__init__()
    # store edge_index and edge_weight as buffers so that they stay fixed
    # bc values stay same for every cell line
    self.register_buffer("edge_index", edge_index)
    self.register_buffer("edge_weight", edge_weight)

    # create gene embeddings
    # a Parameter of shape [n_genes, emb],
    # small random values so that gene values can be passed in
    self.gene_emb = nn.Parameter(torch.randn(n_genes, emb) * 0.1)

    # two layers of GCN
    self.conv1 = GCNConv(emb+1, hidden)
    self.conv2 = GCNConv(hidden, hidden)

    # dropout layer:
    # randomly zeroes 30% of numbers so model cant rely on one, fights overfitting
    self.drop = nn.Dropout(dropout)

    # output layer: combine all genes features into one prediction per cell line
    self.out = nn.Linear(n_genes * hidden, 1)

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
    h = torch.cat([h, e], dim = -1)

    # each gene takes weighted sum of itself and its neighbors (learning), 16 learned features
    # ReLU replaces negative numbers with 0
    h = torch.relu(self.conv1(h, self.edge_index, self.edge_weight))

    # repeat, information is now two steps away
    h = torch.relu(self.conv2(h, self.edge_index, self.edge_weight))

    # flatten, dropout, output
    h = h.flatten(1)
    h = self.drop(h)
    pred = self.out(h)

    return pred



if __name__ == "__main__":
  from graph import load_gene_order, load_edges, build_edge_index
  genes, idx = load_gene_order("data/fake/CellLinesLandmarkGenes.csv")
  edges = load_edges("data/fake/curated_interaction_edges.json", idx)
  edge_index, edge_weight = build_edge_index(edges)

  # ---- verify shape of outputs ----
  model = GeneGCN(edge_index, edge_weight, n_genes=len(genes))
  x = torch.randn(4, len(genes))
  out = model(x)
  print(out.shape)

  # ---- memorization test ----
  import torch.nn.functional as F

  torch.manual_seed(0)
  x_small = torch.randn(20, len(genes))
  y_small = torch.randn(20)

  model = GeneGCN(edge_index, edge_weight, n_genes=len(genes), dropout=0.0)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  model.train()
  for step in range(500):
      pred = model(x_small).squeeze(-1)       # predict
      loss = F.mse_loss(pred, y_small)        # measure error
      optimizer.zero_grad()                   # clear old gradients
      loss.backward()                         # gradient calculations
      optimizer.step()                        # apply gradients,move the weights
      if step % 50 == 0:
          print(f"step {step:3d}   loss {loss.item():.4f}")

