# -*- coding: utf-8 -*-

import csv
import numpy as np

import matplotlib.pyplot as plt


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
        data = np.array(data, dtype=np.float)
        labels = np.array(labels, dtype=np.float)

    # return data with the associated label
    return data, labels


def plot_dataset2d(data, labels, theta=None):
    x_pos = data[(labels == 1),:]
    x_neg = data[(labels ==  0),:]

    #plt.plot(x_pos[:,1],x_pos[:,2], marker = '+' )
    #plt.plot(x_neg[:,1],x_neg[:,2], marker='o')
    plt.scatter(x_pos[:,1],x_pos[:,2], marker = '+' , label = "pos")
    plt.scatter(x_neg[:,1],x_neg[:,2], marker='o', label = "neg")
    
    plt.xlabel("x1")
    plt.ylabel("x2")

    if theta is not None:
        y30 = - 1/theta[2]*(theta[0]+30*theta[1])
        y100 = - 1/theta[2]*(theta[0]+100*theta[1])
        plt.plot([30, 100], [y30, y100], '-r')
    plt.legend()
    plt.show()


def plot_loss(h):
    plt.plot(h)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()