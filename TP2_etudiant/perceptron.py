import csv
import shutil

import matplotlib.pyplot as plt
import numpy as np

from utils import plot_training


class Perceptron(object):
    """ Classe codant le fonctionnement du perceptron
    dans sa version non stochastique
    """
    def __init__(self, in_features:int, learning_rate:float=0.01, lr_decay:bool=False, max_iter:int=200,
                 early_stopping:bool=False, tol:float=1e-6, display:bool=False) -> None:
        # paramètres générals de la classe
        self.in_features = in_features              # taille d'entrée du perceptron
        self.lr = learning_rate                     # taux d'apprentissage
        self.lr_decay = lr_decay                    # modifie le taux d'apprentissage à chaque itération
                                                    # en le divisant par le nombre d'itération déjà passée
        self.max_iter = max_iter                    # nombre d'epoch
        self.early_stopping = early_stopping        # arrête l'apprentissage si les poids
                                                    # ne s'améliorent pas
        self.tol = tol                              # différence entre avant et après la
                                                    # mise à jour des poids
        self.display = display                      # affichage de l'apprentissage du perceptron

        # initialisation quelconques des connexions synaptiques
        # on considèrera le biais comme la multiplication d'une entrée de valeur 1.0 par un poids associé
        # le biais est utilisé comme seuil de décision du perceptron lors de la prédiction
        #self.weights = np.array([1, -0.8, 0.5])
        #self.weights = np.array([1., -1., 0.0])
        #self.weights = np.array([-2, -3, 6.0])
        self.weights = np.array([0.1, -0.2, 0.3])
        #self.weights = np.random.normal(0, 1, size=in_features+1)

    def predict(self, X:np.ndarray) -> np.ndarray:
        """Prédiction des données d'entrée par le perceptron

        X est de la frome [nb_data, nb_param]
        La valeur renvoyée est un tableau contenant les prédictions des valeurs de X de la forme [nb_data]
        """
        #print(self.weights[1:].shape)
        #temp=(self.weights[1:]*X).reshape((len(X),2)).sum(axis=1)+self.weights[0]
        temp=np.dot(X,self.weights[:self.in_features])+1*self.weights[self.in_features]
        temp=temp>0
        temp = np.where(temp > 0, 1, -1)
        
        #print(temp)
        return temp
    
    def fit(self, X:np.ndarray, y:np.ndarray) -> np.ndarray:
        """Apprentissage du modèle du perceptron

        X : données d'entrée de la forme [nb_data, nb_param]
        y : label associée à X ayant comme valeur
                 1 pour la classe positive
                -1 pour la classe négative
            y est de la forme [nb_data]
        """
        # vérification des labels
        assert np.all(np.unique(y) == np.array([-1, 1]))

        # Sauvegarde tous les calculs de la somme des distances euclidiennes pour l'affichage
        if self.display:
            shutil.rmtree('./img_training', ignore_errors=True)
            metric = []
        
        # initialisation d'un paramètre permettant de stopper les itérations lors de la convergence
        stabilise = False
        lr = self.lr
        # apprentissage sur les données
        errors = np.zeros(self.max_iter)
        for iteration in range(self.max_iter):
            # variable stockant l'accumulation des coordonnées
            modif_w = np.zeros(len(self.weights))
            
            prediction=self.predict(X)
            test=0
            for point, label in zip(X, y):
                # prédiction du point
                ...
                #print(point)
                predict_pt=self.predict(point)
                # accumulation des coordonnées suivant la classe si les données sont mal classées
                if predict_pt!=label:
                    errors[iteration] += 1
                    modif_w +=label*np.append(point,np.array([1.]))
                    #mettre a jour le biais
                    test+=1
            print("iteration ", iteration, " :",  test , " points mal classés")
            # affichage de l'erreur et de la ligne séparatrice
            if self.display:
                plot_training(iteration, X, y, self.weights, list(errors[:iteration+1]))
            
            # mise à jour des poids
            old_weights = np.array(self.weights)
            # if self.lr_decay:
            #     lr = self.lr/(iteration+1)
            # else:
            #     lr = self.lr
                
                
            if self.lr_decay:
                lr = lr*0.99
            else:
                lr = self.lr
            
            self.weights += lr*modif_w
            print(self.weights)
            # stopper l'algorithme lorsque l'algorithme converge
            if self.early_stopping:
                # A compléter
                
                stabilise = np.sum(np.absolute(self.weights-old_weights)/old_weights)< self.tol or test==0
                
                #stabilise=np.all(prediction==label)
                
                stabilise= test==0
                if stabilise:
                    # on affiche le dernier hyperplan calculé
                    plot_training(iteration, X, y, self.weights, list(errors[:iteration+1]))
                    # on arrete l'apprentissage
                    break

    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Retourne la moyenne de précision sur les données de test et labels
        """
        return np.sum(y == self.predict(X)) / len(X)
