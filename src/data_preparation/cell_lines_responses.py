import pandas as pd
from pathlib import Path

path_to_raw_data = Path("raw_datasets")
path_to_curated_data = Path("curated_data")

# Gets a curated gene panel for each cell line, which includes landmark genes and drug targets.
def mapCellLinesToTargetGenes():
    # Get raw data
    response_df = pd.read_excel(path_to_raw_data / "GDSC2_fitted_dose_response_27Oct23.xlsx")
    landmark_df = pd.read_csv(path_to_raw_data / "LandmarkGenes.txt", sep="\t")
    cell_line_df = pd.read_csv(path_to_raw_data / "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    
    # Keep only landmark
    curated_landmark_df = landmark_df[landmark_df['Type'] == 'landmark']
    gene_symbols = set(landmark_df["Symbol"])
    curated_gene_symbols = set(curated_landmark_df["Symbol"])
    print(f"Number of genes from landmark set: {len(curated_gene_symbols)}")

    
    # Also get the putative target genes
    putative_target_genes = set()
    exempt_targets = set()
    for index, item in response_df.iterrows():
        val = item['PUTATIVE_TARGET']
        if isinstance(val, str):
            targets = val.strip().split(",")
            for target in targets:
                # assumes gene_symbols contains every gene symbol we have
                if target in gene_symbols:
                    putative_target_genes.add(target)
                else:
                    exempt_targets.add(target)
                    
    print(f"Found {len(exempt_targets)} exempt targets")    
    curated_gene_symbols = curated_gene_symbols.union(putative_target_genes)
    print(f"Number of genes from landmark set + putative targets: {len(curated_gene_symbols)}")
    
    drop_cols = []
    for index, col in enumerate(cell_line_df.columns):
        if index > 0:
            # Columns look like "BRAF (12345)"
            if col.strip().split()[0] not in curated_gene_symbols:
                drop_cols.append(col)
    
    cell_line_df = cell_line_df.drop(columns=drop_cols)
    print(f"Result is dataframe with {cell_line_df.shape[0]} items and {cell_line_df.shape[1]} genes")
    cell_line_df.to_csv(path_to_curated_data / "CellLinesLandmarkGenes.csv", index=False)

# Maps the gene panel of a cell line to a drug and the respective lnc50 concentration
# of that drug needed to kill the cell.
def mapCellLineTolnc50():
    
    response_df = pd.read_excel(path_to_raw_data / "GDSC2_fitted_dose_response_27Oct23.xlsx")
    cell_line_df = pd.read_csv(path_to_curated_data / "CellLinesLandmarkGenes.csv")
    model_df = pd.read_csv(path_to_raw_data / "model.csv")
    
    response_model_merge = pd.merge(response_df, model_df, left_on="SANGER_MODEL_ID", right_on="SangerModelID").drop(columns=["SangerModelID"])
    response_cell_line_merge = pd.merge(cell_line_df,response_model_merge, left_on=cell_line_df.columns[0], right_on="ModelID")
    
    print(f"Result is dataframe with {response_cell_line_merge.shape[0]} items and {response_cell_line_merge.shape[1]} columns")
    response_cell_line_merge.to_parquet(path_to_curated_data / "ResponseCellLineMerge.parquet", index=False)
