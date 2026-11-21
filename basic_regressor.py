import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr


# builds ridge and dummy regressor and puts vals in a dict
def build_models():
    result={}
    alphas=np.logspace(-2,2,5)
    dummy=DummyRegressor(strategy='mean')
    ridge=make_pipeline(MinMaxScaler(),RidgeCV(alphas=alphas))
    result['dummy']=dummy
    result['ridge']=ridge
    return result

# for each drug
def run_drug(X,y,folds):
    # 5 fold cross val
    results=[]
    # for ridge
    coefs=[]
    # to match up train and test set with other models
    for i,(train,tests) in enumerate(PredefinedSplit(test_fold=folds).split()):
        models=build_models()
        X_train=X.iloc[train]
        X_tests=X.iloc[tests]
        y_train=y.iloc[train]
        y_tests=y.iloc[tests]
        for name, model in models.items():
            model.fit(X_train,y_train)
            preds=model.predict(X_tests)
            rmse=root_mean_squared_error(y_tests,preds)
            r2=r2_score(y_tests,preds)
            spearman=spearmanr(y_tests,preds)[0]
            # results holds metrics
            results.append({"model":name, "fold":i, "rmse":rmse,"r2":r2,"spearman":spearman })
            # coefs keeps all ridge data
            if name=="ridge":
                coefs.append({"fold":i, "alpha":model[-1].alpha_,"weights":pd.Series(model[-1].coef_,index=X.columns), "bias": model[-1].intercept_})
    return results,coefs
            
# all drugs are tested 
def all_drugs(data):
    results=[]
    coefs=[]
    for drug, (X,y,folds) in data.items():
        drugresults,drugcoefs=run_drug(X,y,folds)
        for row in drugresults:
            row["drug"]=drug
        for rows in drugcoefs:
            rows["drug"]=drug
        results.extend(drugresults)
        coefs.extend(drugcoefs)

    return results,coefs


def rank_genes(drugcoefs):
    # stack fold weights (5, n_gene)
    weights=pd.DataFrame([row["weights"] for row in drugcoefs])

    # average across folds then absolute values so if direction flips across folds get marked down
    meanweight=weights.mean(axis=0).abs()

    # most to least important
    # names in order and score
    return meanweight.sort_values(ascending=False)


# loads data for ridge and dummy
def load_data():
    folds=pd.read_csv('data/fold_assignments.csv')
    response=pd.read_csv('data/curated_data/ResponseCellLineMerge.csv')
    expression=pd.read_csv('data/curated_data/CellLinesLandmarkGenes.csv',index_col=0).rename(columns=lambda c: c.split(" (")[0])

    data={}
    # each unique drug id
    for i in folds["DRUG_ID"].unique():
        folds_d=folds[folds["DRUG_ID"]==i]
        response_d=response[response["DRUG_ID"]==i]
        # merge folds and response by modelid
        merger=folds_d.merge(response_d[["ModelID","LN_IC50"]], on="ModelID",how="inner")
        X=expression.loc[merger["ModelID"]]
        y=merger["LN_IC50"]
        folds_arr=merger["fold"]
        data[i]=(X,y,folds_arr)
        
    return data

# checks if target gene in top 10 25 50 
def targethit(ranked,target,k=(10,25,50)):
    hit={}
    for i in k:
        topk=set(ranked.index[:i])
        hit[f"hit@{i}"]=any(t in topk for t in target)
    return hit


def main():
    data=load_data()
    # our control grp
    controls=pd.read_csv("data/drug_list_preliminary_controls.csv")
    target=dict(zip(controls["drug"],controls["targets"].str.split(", ")))
    data={d:data[d]for d in target}
    results,coefs=all_drugs(data)
    # rq1
    results_df=pd.DataFrame(results)
    results_df.to_csv("data/results/baseline.csv",index=False)
    # rq2
    recovery=[]
    for d in data:
        drugcoefs=[row for row in coefs if row["drug"]==d]
        ranked=rank_genes(drugcoefs)
        row={"drug":d,"targets":"/".join(target[d])}
        row.update(targethit(ranked,target[d]))
        recovery.append(row)

    recover_df=pd.DataFrame(recovery).merge(controls[["drug","name","control_label"]],on="drug")
    recover_df.to_csv("data/results/ridge_target_recovery.csv",index=False)


    print("Saved", len(results_df),"rows to baseline.csv")
    print("Saved", len(recover_df),"rows to ridge_target_recovery.csv")

if __name__=="__main__":
    main()

        
    