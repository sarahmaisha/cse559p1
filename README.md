# cse559p1
Project 1 for CSE 559 Team 6

run on python 3.12!!


uv add torch torch-geometric shap scikit-learn scipy

to get fake data:
cd to root
uv run python src/gnn/fake_data.py

full steps to run
# 1. Import tar data into data/
# 2. Change paths in prepare_real.py, run_gnn.py
need to eventually change to argument for terminal line?

# 1. make responses.csv, folds.csv, drug_list.csv in the v2 folder
uv run python src/gnn/prepare_real.py

# 2. train all 6 drugs x 5 folds (~4-5 min)
uv run python src/gnn/run_gnn.py

# 3. summary tables
uv run python src/gnn/summarize.py results/predictions/gnn_real_v2.csv


