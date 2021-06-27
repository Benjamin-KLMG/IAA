import numpy as np

from bayes import GaussianBayes
from utils import load_dataset, plot_scatter_hist


def main():
    train_data, train_labels = load_dataset("./data/train.csv")
    test_data, test_labels = load_dataset("./data/test.csv")

    # affichage
    ...
    plot_scatter_hist(train_data,train_labels)
    test=train_labels.reshape((len(train_labels),1))
    
    #print(train_labels)
    #print(np.concatenate((train_data,test),axis=1 ))
    #data=np.concatenate((train_data,test),axis=1 )
    #np.mean(data==0.)
    # Instanciation de la classe GaussianB
    g = GaussianBayes(priors=[1./3,1./3,1./3,])
    #g = GaussianBayes(priors=[0.8, 0.1, 0.1])
        
    # Apprentissage
    
    
    g.fit(train_data, train_labels)
    #print(g.predict(train_data))
    # Score
    score = g.score(test_data, test_labels)
    print("precision : {:.2f}".format(score))

    # input("Press any key to exit...")


if __name__ == "__main__":
    main()
