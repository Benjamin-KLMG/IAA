import numpy as np

from Donnees import lire_image,pretraitement,moyenne_images,plot_training,affichage_images,affichage_Vraisemblance
from bayes import GaussianBayes
#from utils import load_dataset, plot_scatter_hist


def main():
    affichage_images(type_fleur='ch', n_fleurs=10)
    image_vect = lire_image("Fleurs/pe1.png")

    # affichage
    #print(image_vect.shape)
    rv=pretraitement(image_vect)
    #print(rv)
    #On recupere toute les donnee pour faire lapprentissage des parametre
    data=moyenne_images()
    
    plot_training(data)
    affichage_Vraisemblance(data)
    # Instanciation de la classe GaussianB
    g = GaussianBayes()
    g = GaussianBayes(priors=[0.8, 0.1, 0.1])
        
    # Apprentissage
    
    train_labels=["pe","ch","oe"]
    g.fit(data, train_labels)
    #print(g.sigma)
    print(g.predict(rv.reshape((1,2))))
    print(f"la feuille ch1 est predit pour la classe : {train_labels[g.predict(rv.reshape((1,2)))[0]]} ")


    # input("Press any key to exit...")


if __name__ == "__main__":
    main()
