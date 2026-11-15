import networkx as nx
import pandas as pd
from pathlib import Path
from itertools import combinations
import json

path_to_raw_data = Path("raw_datasets")
path_to_cell_line_graphs = Path("graphs")
path_to_curated_data = Path("curated_data")
INTERACTION_CUTOFF = 0.750

# Constructs the cell line graph from disk
def construct_graph(sanger_id) -> nx.Graph:
    cell_graph = nx.read_graphml(path_to_cell_line_graphs / f"{sanger_id}.graphml")
    with open(path_to_curated_data / "curated_interaction_edges.json", "r") as file:
        edge_list = json.load(file)
        cell_graph.add_weighted_edges_from(edge_list)
    return cell_graph

# Returns normalized value for transcripts per million
def normalize_tpm_value(tpm_value, tpm_max, tpm_min):
    return (tpm_value - tpm_min) / (tpm_max - tpm_min)

# Returns float value of ppi score
def normalize_interaction_score(score):
    return score / 1000.0

# writes a list of interactions greater than the cutoff. Scores are renormalized
def write_curated_interaction_list():
    # Read edge list from file, only include edges that meet the cutoff.
    with open(path_to_curated_data / "interaction_edges.json", "r") as file:
            interaction_list = json.load(file)
            simplified_interaction_list = []
            for interaction in interaction_list:
                score = interaction[2]["combined_score"]
                if score > INTERACTION_CUTOFF:
                    renormalized_score = round((score - INTERACTION_CUTOFF) / (1.0 - INTERACTION_CUTOFF),3)
                    simplified_interaction_list.append((interaction[0],interaction[1], renormalized_score))
        
    with open(path_to_curated_data / "curated_interaction_edges.json", "w") as file:
        json.dump(simplified_interaction_list, file, indent=4)

# Constructs a network from a gene panel and interaction list. Writes to file
def write_gene_protein_network(sanger_id, genes):
    gene_graph = nx.Graph()
    gene_graph.add_nodes_from(genes)
    nx.write_graphml(gene_graph, path_to_cell_line_graphs / f"{sanger_id}.graphml")

# Constructs all gene-protein networks from a list of cell lines
def make_gene_protein_networks():
    response_cell_line_df = pd.read_parquet(path_to_curated_data / "ResponseCellLineMerge.parquet")
    cell_lines = pd.read_csv(path_to_curated_data / "CellLinesLandmarkGenes.csv")
    tpm_max = cell_lines.max(numeric_only=True).max()
    tpm_min = cell_lines.min(numeric_only=True).min()
    for _,row in cell_lines.iterrows():
        cell_line_response = response_cell_line_df.loc[response_cell_line_df["ModelID"] == row.iloc[0], 'SANGER_MODEL_ID']
        if not cell_line_response.empty:
            sanger_id = cell_line_response.iloc[0]
            # Get the node values from gene tpm
            normalized_gene_scores = []
            genes = row.iloc[1:]
            for gene, tpm in genes.items():
                norm_tpm = normalize_tpm_value(tpm, tpm_max, tpm_min)
                normalized_gene_scores.append((gene, {"tpm_norm" : round(norm_tpm,3)}))

            write_gene_protein_network(sanger_id, normalized_gene_scores)
             
# Builds an interaction list of genes by taking isoform protein interactions an mapping it to 
# genes from the curated gene panel.
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
            
# Replaces protein IDs in interaction dataset with preferred names, which maps to gene panel.  
def interaction_scores_preferred_names():
    
    protein_info = pd.read_csv(path_to_raw_data / "9606.protein.info.v12.0.txt",sep="\t")

    protein_interactions = pd.read_csv(path_to_raw_data / "9606.protein.links.v12.5.txt",sep=r"\s+")

    id_to_preferred = protein_info.set_index("#string_protein_id")["preferred_name"]

    protein_interactions["protein1"] = protein_interactions["protein1"].map(id_to_preferred)
    protein_interactions["protein2"] = protein_interactions["protein2"].map(id_to_preferred)

    protein_interactions.to_csv(path_to_curated_data / "preferred.name.interactions.txt",sep=" ",index=False)