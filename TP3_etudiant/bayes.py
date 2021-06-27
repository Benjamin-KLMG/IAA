import numpy as np
from typing import Union
from math import sqrt, pi, exp , log


class GaussianBayes(object):
    """ Classification by normal law by Bayesian approach
    """
    def __init__(self, priors:np.ndarray=None) -> None:
        self.priors = priors    # a priori probabilities of classes
                                # (n_classes,)

        self.mu = None          #  mean of each feature per class
                                # (n_classes, n_features)
        self.sigma = None       # covariance of each feature per class
                                # (n_classes, n_features, n_features)
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        X shape = [n_samples, n_features]
        maximum log-likelihood
        """
        n_obs = X.shape[0]
        n_classes = self.mu.shape[0]
        n_features = self.mu.shape[1]

        # initalize the output vector
        y = np.empty(n_obs,dtype=int)
        #d=np.linalg.det()
        c_=np.zeros(n_classes)
        a=np.zeros((n_features,n_features))
        for j,x in enumerate(X):
            for i in range(n_classes):
                diag=self.sigma[i].diagonal()
                np.fill_diagonal(a,diag)
                #print(a,i)
                #sig_inv=np.linalg.inv(self.sigma[i]) #l'nverse de la matrice sigma
                sig_inv=np.linalg.inv(a)
                x_u=x - self.mu[i] 
                #c_[i]=-((n_features/2.)*log(2*pi) + 0.5*log(np.linalg.det(self.sigma[i]))) - 0.5*((x_u.T).dot(sig_inv)).dot(x_u)
                c_[i]=-0.5*log(np.linalg.det(self.sigma[i])) - 0.5*((x_u.T).dot(sig_inv)).dot(x_u) + log(self.priors[i])
            y[j]=np.argmax(c_)
            
            #print(y)
        return y
    
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """Learning of parameters
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        # number of random variables and classes
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        # initialization of parameters
        self.mu = np.zeros((n_classes, n_features))
        self.sigma = np.zeros((n_classes, n_features, n_features))
        
        # learning        
        for i in range(n_classes):
            self.mu[i]=np.mean(X[(y==i)],axis=0 ) #la moyenne de chaque classe de donne
            self.sigma[i]=np.cov(X[ (y==i)].T) #la matrice de cov pour chaque classe de donne
        
        #self.predict(X)
        

    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Compute the precision
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        return np.sum(y == self.predict(X)) / len(X)
