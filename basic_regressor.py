import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import root_mean_squared_error, r2_score
from scipy.stats import spearmanr

# builds ridge and dummy regressor and puts vals in a dict
def build_models():
    result={}
    alphas=np.logspace(-2,2,5)
    dummy=DummyRegressor(strategy='mean')
    ridge=RidgeCV(alphas=alphas)
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
        X_train=X[train]
        X_tests=X[tests]
        y_train=y[train]
        y_tests=y[tests]
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
                coefs.append({"fold":i, "alpha":model.alpha_,"weights":model.coef_, "bias": model.intercept_})
    
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






    



        
    