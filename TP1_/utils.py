# -*- coding: utf-8 -*-

import csv
import os
# from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def check_compute_distances(kmeans, mat):

    vec = np.random.normal(size=mat.shape[1])
    d = kmeans._compute_distances(mat, vec)
    OK_dim = d.shape[0] == mat.shape[0]
    if not OK_dim:
        print("Erreur de dimension: _compute_distances doit renvoyer un vecteur de taille le nombre d'exemples, ici (%d,)"%mat.shape[0])

    correct = np.linalg.norm(mat - vec, axis=-1) ** 2

    OK_calcul = np.allclose(d, correct)
    if not OK_calcul:
        print("Erreur de formule: revoir le calcul fait dans _compute_distances")

    if OK_dim and OK_calcul: print("_compute_distances est OK ! Bravo")


def load_dataset(pathname:str):
    """Load a dataset in csv format.

    Each line of the csv file represents a data from our dataset and each
    column represents the parameters.
    The last column corresponds to the label associated with our data.

    Parameters
    ----------
    pathname : str
        The path of the csv file.

    Returns
    -------
    data : ndarray
        All data in the database.
    labels : ndarray
        Labels associated with the data.
    """
    # check the file format through its extension
    if pathname[-4:] != '.csv':
        raise OSError("The dataset must be in csv format")
    # open the file in read mode
    with open(pathname, 'r') as csvfile:
        # create the reader object in order to parse the data file
        reader = csv.reader(csvfile, delimiter=',')
        # extract the data and the associated label
        # (he last column of the file corresponds to the label)
        data = []
        labels = []
        for row in reader:
            data.append(row[:-1])
            labels.append(row[-1])
        # converts Python lists into NumPy matrices
        # in the case of the list of labels, generate an int id per class
        data = np.array(data, dtype=np.float)
        lookupTable, labels = np.unique(labels, return_inverse=True)
    # return data with the associated label
    return data, labels


def plot_training(iteration:int, data:np.ndarray, labels:np.ndarray, cluster_centers:np.ndarray,
                  metric:list=None, figure:int=1, save_png:bool=False) -> None:
    if len(cluster_centers) > 12:
        raise Exception("L'affichage ne gère pas plus de 12 clusters...")

    customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', '#04AF00', '#39A7FF',
                     '#7519CC', '#79E7FF', '#1863C15', '#B72EB9', '#EC2328', '#C86D39']
    fig = plt.figure(figure, clear=True)

    # nombre de dimensions
    n_dim = data.shape[1]
    n_subplot = 2 if metric else 0
    #print("heee hooooo",n_dim)
    if n_dim == 2:
        #print("heee hooooo")
        ax = fig.add_subplot(1, n_subplot, 1)
        ax.set_title("Iteration {}".format(iteration))
        for y in np.unique(labels):
            x = data[labels == y]
            # add data points
            ax.scatter(x=x[:,0],
                       y=x[:,1],
                       alpha=0.20,
                       color=customPalette[int(y)])
            # add label
            ax.annotate(int(y),
                        cluster_centers[int(y)],
                        horizontalalignment='center',
                        verticalalignment='center',
                        size=20, weight='bold',
                        color=customPalette[int(y)])
    
    else:
        print("heee hooooo")
        ax = fig.add_subplot(1, n_subplot, 1, projection='3d')
        ax.set_title("Iteration {}".format(iteration))
        for y in np.unique(labels):
            x = data[labels == y]
            # add data points
            ax.scatter(x[:,0], x[:,1], x[:,2],
                        alpha=0.20,
                        color=customPalette[int(y)])
            # add label
            ax.text(cluster_centers[int(y),0],
                    cluster_centers[int(y),1],
                    cluster_centers[int(y),2],
                    int(y),
                    size=20, weight='bold', zorder=1,
                    color=customPalette[int(y)])
    
    if metric:
        ax = fig.add_subplot(1, n_subplot, 2)
        ax.set_title("différence SSE entre 2 itérations")
        plt.plot(metric, '-', color=customPalette[-1])
    
    if save_png:
        os.makedirs('./img_training', exist_ok=True)
        plt.savefig("./img_training/im{}.png".format(iteration))
    # met à jour l'affichage
    plt.pause(0.25)
