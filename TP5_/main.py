# -*- coding: utf-8 -*-

# author: Thomas Pellegrini

import numpy as np

from logreg import Logreg, cout_fn
from utils import load_dataset, plot_dataset2d, plot_loss


if __name__ == '__main__':

    csv_chemin = 'data/data1_logreg.csv'

    data, labels = load_dataset(csv_chemin)

    m, d = data.shape

    #print(data.shape, labels.shape)
   

    # ajouter un 1 à tous les samples X pour gérer le biais dans theta
    data = np.concatenate(( np.ones((m,1)) , data) , axis = 1)
   # print(data)

    plot_dataset2d(data, labels, theta=None)
    
    # Initialisation de theta avec un vecteur de zéros
    theta = np.zeros((d+1))
    #print(theta.T)

    use_momentum = True
    # use_momentum = True
    if use_momentum:
        lr = 0.0015
        max_iter = 1000
    else:
        lr = 0.0005
        max_iter=200000

    model = Logreg(max_iter=max_iter, theta=theta, lr=lr, use_momentum=use_momentum, display=False,gama=0.99)
    #print(model.sigmoid(np.array([[0.4 ,0.3, 0.11], [0.224 ,0.3656, 0.11]])))

    pred = model.predict(data)

    #print(pred.shape)
    sig = model.sigmoid( data @ model.theta)
    pred = model.predict(data)
    #print(pred)
    #c_init = cout_fn(data, sig, labels)
    c_init, grad_init = cout_fn(data, sig, labels)
    print("Valeur coût initial : %.3f" % c_init)
    print("grad_init", grad_init)
    
    s = model.score(data, labels)
    print("avant apprentissage, acc=%.2f"%s)

    h = model.fit(data, labels)
    #print(h)
    plot_loss(h)

    s = model.score(data, labels)
    print("après apprentissage, acc=%.2f"%s)
    # 200000 it, apprentissage, acc = 0.91

    plot_dataset2d(data, labels, theta=model.theta)
    