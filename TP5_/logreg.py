# -*- coding: utf-8 -*-

# author: Thomas Pellegrini

import numpy as np
from math import exp

def cout_fn(data: np.ndarray, pred: np.ndarray, labels: np.ndarray):
    
    cout = (1 / data.shape[0]) * ((- labels @ np.log(pred) ) - (1 - labels)@ np.log(1 - pred)).sum()

    grad = 1 / data.shape[0] * ((pred - labels)@data)
    
    return cout ,grad


class Logreg(object):

    def __init__(self, max_iter: int, theta: np.ndarray = None, lr: float = 0.005,
                 use_momentum: bool = False, display: bool = False,gama=0.99 ) -> None:

        self.max_iter = max_iter                # Nombre d'itérations
        self.display = display                  # affichage dans fit()
        self.theta = theta
        self.lr = lr
        self.use_momentum = use_momentum
        self.gama=gama
    
    def sigmoid(self, Z):
        return (1 / (1 + np.exp(- Z)))
    

    def predict(self, X: np.ndarray) -> np.ndarray:
        """inférence: retourne les probabilités de la classe positive
        """
        #sig = self.sigmoid((self.theta.T).dot(X.T))
        sig = self.sigmoid( X @ self.theta)
        #predictions= np.where(sig > 0.5 , 1 , 0)
        return sig
      

    def fit(self, X: np.ndarray, y: np.ndarray) -> list:
        """Apprentissage des poids theta:
        - par descente de gradient si use_momentum est False
        - par descente de gradient avec momentum dans le cas contraire
        Retourne la liste des coûts obtenus à chaque epoch
        """
        list_cout=[]
        list_theta=[]
        v=0
        for i in range(self.max_iter):
            
            cout,grad=cout_fn(X,self.predict(X) , y)
            list_cout.append(cout)
            list_theta.append(self.theta)
            #print(cout)
            if self.use_momentum:
                v=self.gama*v + self.lr*grad
                self.theta=self.theta - v
            else:
                self.theta=self.theta-self.lr*grad
            #print("tetha : ",self.theta)
        #self.theta=list_theta[np.argmin(np.array(list_cout))]
        return list_cout 


    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calcule l'accuracy
        """
        n_data = X.shape[0]

        y_pred = self.predict(X)

        score = np.sum(1*(y_pred>0.5) == y)/n_data

        return score
