import numpy as np
import matplotlib.pyplot as plt


# always predicts the training-set mean response
class DummyRegressor():

    def __init__(self, trainmean=0):
        self.trainmean=trainmean

    def fit(self,x,true,iterations=None):
        self.trainmean=true.mean()
   
    def predict(self, x):
        return np.full(len(x),self.trainmean)
        
    