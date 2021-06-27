# -*- coding: utf-8 -*-

# author: Benjamin Chamand, Thomas Pellegrini

import shutil

import numpy as np

from utils import plot_training


class KMeans(object):
    def __init__(self, n_clusters:int, max_iter:int, early_stopping:bool=False,
                 tol:float=1e-4, display:bool=False) -> None:
        self.n_clusters = n_clusters            # Nombre de clusters
        self.max_iter = max_iter                # Nombre d'itérations
        self.early_stopping = early_stopping    # arrête l'apprentissage si 
        self.tol = tol                          # seuil de tolérance entre 2 itérations
        self.display = display                  # affichage des données

        self.cluster_centers = None             # Coordonnées des centres des clusters
                                                # (centre de gravité des classes)
    
    def _compute_distances(self, v1:np.ndarray, v2:np.ndarray) ->  np.ndarray:
        """Retourne les distances quadratiques entre les arrays v1 et v2, de dimensions quelconques (squared euclidian distance)
        """
        # A faire en une seule ligne de code si possible (pas de boucle)
        #return ((v1  - v2)**2).sum(axis = 1)
        return np.linalg.norm(v1 - v2, axis=-1) ** 2
       # pass
    
    def _compute_inertia(self, X:np.ndarray, y:np.ndarray) -> float:
        """Retourne la Sum of Squared Errors entre les points et le centre de leur
        cluster associe
        """
        #dict = {index:x for index, x in enumerate(self.cluster_centers) }
        somme=0
        for i in range(len(X)):
            # '''On parcours les valeurs de X en calculant la difference de chaque point
            # avec le centroide de la classe correspondant a ce point
            # en utilisan la methode _compute_distances 
            # '''
            somme+=self._compute_distances(X[i],self.cluster_centers[y[i]])
            #print(X[i],self.cluster_centers[y[i]])
            
        return somme
        

    
    def _update_centers(self, X:np.ndarray, y:np.ndarray) -> None:
        """Recalcule les coordonnées des centres des clusters
        """
        count=np.zeros(self.n_clusters)
        centers=np.zeros((self.n_clusters,len(X[0])))
        for i,val in  enumerate(X):
            centers[y[i]]+=val
            count[y[i]]+=1
            
        self.cluster_centers=centers/count.reshape(self.n_clusters,1)
        
        


    
    def predict(self, X:np.ndarray) -> np.ndarray:
        """attribue un indice de cluster à chaque point de data

        X = données
        y = cluster associé à chaque donnée
        """
        # nombre d'échantillons
        n_data = X.shape[0]
        y = np.zeros(n_data, dtype=int)
        for i in  range(n_data):
            dist = self._compute_distances(X[i],self.cluster_centers)
            y[i] = np.argmin(dist)
            

        return y


    def fit(self, X:np.ndarray) -> None:
        """Apprentissage des centroides
        """
        # Récupère le nombre de données
        n_data = X.shape[0]

        # Sauvegarde tous les calculs de la somme des distances euclidiennes pour l'affichage
        if self.display:
            shutil.rmtree('./img_training', ignore_errors=True)
            metric = []

        # 2 cas à traiter : 
        #   - soit le nombre de clusters est supérieur ou égale au nombre de données
        #   - soit le nombre de clusters est inférieur au nombre de données
        if self.n_clusters >= n_data:
            # Initialisation des centroides : chacune des données est le centre d'un clusteur
            self.cluster_centers = np.zeros(self.n_clusters, X.shape[1])
            self.cluster_centers[:n_data] = X
        else:
            # Initialisation des centroides
            #self.cluster_centers = X[:self.n_clusters,:]
            self.cluster_centers = X[np.random.choice(X.shape[0], size=self.n_clusters , replace = False)]

            # initialisation d'un paramètre permettant de stopper les itérations lors de la convergence
            stabilise = False

            # Exécution de l'algorithme sur plusieurs itérations
            for i in range(self.max_iter):
                # détermine le numéro du cluster pour chacune de nos données
                y = self.predict(X)

                # calcule de la somme des distances initialiser le paramètres
                # de la somme des distances
                if i == 0:
                    current_distance = self._compute_inertia(X, y)

                # mise à jour des centroides
                self._update_centers(X, y)
                #print(" nouvau center",self.cluster_centers.shape)

                # mise à jour de la somme des distances
                old_distance = current_distance
                current_distance = self._compute_inertia(X, y)

                # stoppe l'algorithme si la somme des distances quadratiques entre 
                # 2 itérations est inférieur au seuil de tolérance
                if self.early_stopping:
                    # A compléter
                    if old_distance-current_distance < self.tol :
                        stabilise=True
                    
                    #stabilise = ....
                    if stabilise:
                        break

                # affichage des clusters
                if self.display:
                    diff = abs(old_distance - current_distance)
                    metric.append(diff)
                    #print(X)
                    #plot_training(i, X[:,2:], y, self.cluster_centers[:,2:], metric)
                    plot_training(i, X, y, self.cluster_centers, metric)

    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Calcule le score de pureté
        """
        n_data = X.shape[0]

        y_pred = self.predict(X)

        score = 0
        for i in range(self.n_clusters):
            _, counts = np.unique(y[y_pred == i], return_counts=True) 
            score += counts.max()

        score /= n_data

        return score
