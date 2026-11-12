import networkx as nx
import pandas as pd
from pathlib import Path
from itertools import combinations
import json

path_to_raw_data = Path("raw_datasets")
path_to_cell_line_graphs = Path("graphs")
path_to_curated_data = Path("curated_data")
INTERACTION_CUTOFF = 0.750

def normalize_tpm_value(tpm_value, tpm_max, tpm_min):
    return (tpm_value - tpm_min) / (tpm_max - tpm_min)

def normalize_interaction_score(score):
    return score / 1000.0


def make_gene_protein_network(cell_id, genes, interactions):
    gene_graph = nx.Graph()
    gene_graph.add_nodes_from(genes)
    gene_graph.add_edges_from(interactions)
    nx.write_graphml(gene_graph, path_to_cell_line_graphs / f"{cell_id}.graphml")

def make_gene_protein_networks():
    
    cell_lines = pd.read_csv(path_to_curated_data / "CellLinesLandmarkGenes.csv")
    tpm_max = cell_lines.max(numeric_only=True).max()
    tpm_min = cell_lines.min(numeric_only=True).min()
    
    for _,row in cell_lines.iterrows():
        id = row.iloc[0] 
        normalized_gene_scores = []
        genes = row.iloc[1:]
        for gene, tpm in genes.items():
            norm_tpm = normalize_tpm_value(tpm, tpm_max, tpm_min)
            normalized_gene_scores.append((gene, {"tpm_norm" : norm_tpm}))
        
        with open(path_to_curated_data / "interaction_edges.json", "r") as file:
                interaction_list = json.load(file)
                simplified_interaction_list = []
                for interaction in interaction_list:
                    if interaction[2]["combined_score"] > INTERACTION_CUTOFF:
                        simplified_interaction_list.append(interaction)
            
        make_gene_protein_network(id, normalized_gene_scores, simplified_interaction_list)
        break
             

def build_interaction_list():
    
    cell_lines = pd.read_csv(path_to_curated_data / "CellLinesLandmarkGenes.csv")
    protein_interactions = pd.read_csv(path_to_curated_data / "preferred.name.interactions.txt", sep=r"\s+")
    interaction_dict = dict(zip(zip(protein_interactions["protein1"], protein_interactions["protein2"]),protein_interactions["combined_score"]))
        
    gene_names = cell_lines.columns.to_list()[1:]
    interaction_list = []
    
    for gene1, gene2 in list(combinations(gene_names,2)):
        n1 = gene1.strip().split()[0]
        n2 = gene2.strip().split()[0]
        
        if (n1, n2) in interaction_dict or (n1, n2) in interaction_dict:
            interaction_score = interaction_dict[(n1,n2)]
            norm_score = normalize_interaction_score(interaction_score)
            interaction_list.append((gene1, gene2, {"combined_score": norm_score}))
        
    with open(path_to_curated_data / "interaction_edges.json", "w") as file:
        json.dump(interaction_list, file, indent=4)
            
        
def interaction_scores_preferred_names():
    
    protein_info = pd.read_csv(path_to_raw_data / "9606.protein.info.v12.0.txt",sep="\t")

    protein_interactions = pd.read_csv(path_to_raw_data / "9606.protein.links.v12.5.txt",sep=r"\s+")

    id_to_preferred = protein_info.set_index("#string_protein_id")["preferred_name"]

    protein_interactions["protein1"] = protein_interactions["protein1"].map(id_to_preferred)
    protein_interactions["protein2"] = protein_interactions["protein2"].map(id_to_preferred)

    protein_interactions.to_csv(path_to_curated_data / "preferred.name.interactions.txt",sep=" ",index=False)

make_gene_protein_networks()