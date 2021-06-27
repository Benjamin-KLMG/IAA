# -*- coding: utf-8 -*-

# author: Benjamin Chamand, Thomas Pellegrini
import numpy as np
from kmeans import KMeans
from utils import load_dataset, check_compute_distances


def main():
    
    filepath = "./data/self_test.csv"
    #filepath = "./data/iris.csv"

    # chargement des données en deux arrays numpy
    # data : matrice de taille N x d avec N le nb d'exemples et d la dimension des features
    # labels : vecteur de taille N
    data, labels = load_dataset(filepath)
    #print("data",data)
    # initialisation de l'objet KMeans
    kmeans = KMeans(n_clusters=2,
                    max_iter=100,
                    early_stopping=True,
                    tol=1e-6,
                    display=True)
    
    check_compute_distances(kmeans, data)
   
    # calcule les clusters
    
    kmeans.fit(data)

    # calcule la pureté de nos clusters
    score = kmeans.score(data, labels)
    print("Pureté : {}".format(score))

    #input("Press any key to exit...")


if __name__ == "__main__":
    main()
