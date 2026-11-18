import numpy as np
import matplotlib.pyplot as plt

# adapted from assignment 1 linear regression code 
def mse_loss(predicted, true):
    mse = ((true - predicted) ** 2).mean()
    return mse

class RidgeRegression():
    def __init__(self, dim=1, lr=1e-4, lam=1):
        # lr is learning rate of gradient descent 
        self.lr = lr
        self.dim = dim
        self.lam=lam
        # we dont have values for bias and weight so initialize to 0 for now
        self.theta0 = 0
        # one weight per gene, set it to an array of zeroes
        self.theta1 = np.zeros(dim)
        super().__init__()


    def predict(self, x):
        #the equation itself
        # bias + each gene * weight
        return self.theta0 + x@self.theta1
    
    def gradient(self, x, true):
        #partial derivatives of loss function for theta0(bias aka intercept) and theta1(weight aka slope) respectively
        # transpose of x to get one gradient per gene
        return (1/len(true)) * np.sum(self.predict(x) - true), (1/len(true)) * x.T @ (self.predict(x) - true) + (2*self.lam*self.theta1)
   
    def gradient_descent(self, gradients):
        # param-lr*gradient
        return self.theta0 - self.lr * gradients[0], self.theta1 - self.lr * gradients[1]
        

    def fit(self,x,true,iterations):
        loss=[]
        for i in range(iterations):
            loss.append(self.train(x,true))
        return loss


    def train(self, x, true):
        # you can use the gradient and gradient_descent methods to update the parameters
        # return the loss after training
        gradients = self.gradient(x, true)
        self.theta0, self.theta1 = self.gradient_descent(gradients)
        return mse_loss(self.predict(x), true)
        