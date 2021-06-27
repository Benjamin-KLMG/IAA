
import matplotlib.image as img
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import matplotlib.cm as cm
def pretraitement(img):
    
    max_som=np.max(img.sum(axis=2))
    if max_som==0:
        max_som=1
    
    norm=img/max_som

    r = np.mean(norm[:,:,0])
    v = np.mean(norm[:,:,1])
    b = np.mean(norm[:,:,2])
    rv = np.array([r,v]) 
    
    return rv

def moyenne_images(n_fleurs=10):
    cat_fleurs=["pe","ch","oe"]
    feat_dict={"pe":[],"ch":[],"oe":[]}
    for c in cat_fleurs:
        for i in range(n_fleurs):
            path = "./Fleurs/"
            name = path + c + str(i + 1) + ".png"
            print(name)
            img = lire_image(name)
            
            feat_dict[c].append(pretraitement(img))
        feat_dict[c]=np.array(feat_dict[c])
    #print(feat_dict)
    #return feat_dict,
    return np.array([feat_dict['pe'],feat_dict['ch'],feat_dict['oe']])


def lire_image(chemin_fichier):
    return np.array(255*img.imread(chemin_fichier), dtype=int)


def affichage_images(type_fleur='ch', n_fleurs=10):
    """Affichage d'images de fleurs

    Arguments:
        type_fleur='ch', 'oe', 'pe'
        n_fleurs: nombre de fleurs à afficher dans la figure
    """

    for i in range(n_fleurs):
        path = "./Fleurs/"
        name = path + type_fleur + str(i + 1) + ".png"
        print(name)
        image = lire_image(name)
        pretraitement(image)
        plt.figure(1)
        plt.subplot(3, 4, i + 1)
        plt.imshow(image)

    plt.show()




def plot_training( data) -> None:

    customPalette = ['#0015ff', '#ff0000', '#005201']
    markers=['*','.','+']
    plt.figure( clear=True)

    # nombre de dimensions
    n_dim = 2
    #n_subplot = 2 if metric else 1
    n_subplot=1
    cat_fleurs=["pe","ch","oe"]
    cat_num=np.unique(cat_fleurs, return_inverse=True)
    #print("heee hooooo",n_dim)
    if n_dim == 2:
        #print("heee hooooo")
        #ax = fig.add_subplot(1, n_subplot, 1)
        plt.title("Nuage de point ")
        plt.xlabel("Rouge ")
        plt.ylabel("Vert")
        
        for i,x in enumerate(data):
            label=" {} : {}".format(cat_fleurs[i],markers[int(i)])
            #print(label)
            # add data points
            plt.scatter(x=x[:,0],
                       y=x[:,1],
                       alpha=0.20,
                       color=customPalette[int(i)],marker=markers[int(i)], label=label,s=100)
            
        plt.legend()
        plt.show()
            
def affichage_Vraisemblance(X):
    #premier categorie de fleure
    mu_pe=np.mean(X[0],axis=0)
    sig_pe=np.cov(X[0],rowvar=False)
    #print(sig_pe)
    
    #deuxieme ch
    mu_ch=np.mean(X[1],axis=0)
    sig_ch=np.cov(X[1],rowvar=False)
    
    #troisieme oe
    mu_oe=np.mean(X[2],axis=0)
    sig_oe=np.cov(X[2],rowvar=False)

    
        
    def fun(x, mu,sigma):
      return 1./(2*pi*np.sqrt(np.linalg.det(sigma)))*np.exp(-0.5*(x-mu)@np.linalg.inv(sigma)@(x-mu).T)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(0.35, 0.45, 0.001)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(np.array([x,y]),mu_ch,sig_ch) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    
    surf = ax.plot_surface(X, Y, Z,cmap=cm.coolwarm)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    fig.colorbar(surf, shrink=0.5, aspect=5)




if __name__ == '__main__':
    n_fleurs = 10
    affichage_images(type_fleur='pe', n_fleurs=n_fleurs)
    #fit(moyenne_images())
    #moyenne_images()
    plot_training(moyenne_images())
    affichage_Vraisemblance(moyenne_images())