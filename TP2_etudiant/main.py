import argparse

import numpy as np

from perceptron import Perceptron
from utils import load_dataset


def main():
    # chargement des données
    # le fichier .csv contient 3 groupes de points 2D
    # la première colonne du fichier correspond à x1, la deuxième à x2 
    # et la troisième correspond au groupe auquel est associé le point
    filepath = "./data/2Dpoints.csv"
    data, labels = load_dataset(filepath)

    # On garde le groupe de points 1 et 2
    data = data[(labels==0) | (labels==1)]
    labels = labels[(labels==0) | (labels==1)]
    labels = np.where(labels == 0, 1, -1)


    # data = data[(labels==1) | (labels==2)]
    # labels = labels[(labels==1) | (labels==2)]
    # labels = np.where(labels == 1, -1, 1)
    
    print(labels)
    # On garde le groupe de points 1 et 3
    # data = data[(labels==0) | (labels==2)]
    # labels = labels[(labels==0) | (labels==2)]
    # labels = np.where(labels == 0, 1, -1)


    
    # Instanciation de la classe perceptron
    p = Perceptron(2, learning_rate=1e-4, lr_decay=True,
                   early_stopping=True, display=True)

    #a=np.array([3,7])
    #print(a.shape)
    #p.predict(a)
    # Apprentissage
    p.fit(data, labels)

    # Score
    score = p.score(data, labels)
    print("precision : {:.2f}".format(score))

    # input("Press any key to exit...")


if __name__ == "__main__":
    main()
