#Processins of Harshil's millenium data to run on this setup.
import sys
import cPickle as pickle
import pandas as pd
import numpy as np
from sklearn import cross_validation
import gzip

if __name__ == '__main__':


    df = pd.read_csv('/home/mclaughlin6464/GitRepos/ml4cosmosims/data/data.csv')
    #Harshil says I shoudln't have to worry about this.
    '''
    Q = df.values
    M = Q[:,195:201] #mass; need to do this in an iffy way (i.e. not using pandas) because there are duplicate labels

    means = np.mean(M, axis=0)
    stds = np.std(M, axis=0)
    cutoffs = means + 30*stds #cutoffs to remove extreme outliers; 30 seems reasonable

    df = df[df.stellarMass < cutoffs[0]][df.coldGas < cutoffs[1]][df.bulgeMass < cutoffs[2]][df.hotGas < cutoffs[3]][df.blackHoleMass < cutoffs[5]] #god bless pandas
    print df.shape #just to see how many entries we lost; it turns out to be only 48 out of 350k+
    '''

    Q = df.values #same here; this is really not recommended since
    #Normalize by maximum value

    for i in xrange(201):
        if i != 194:
            minval = Q[:,i].min()
            maxval = Q[:, i].max()
            #print val
            Q[:,i]= (Q[:,i]- minval)/(maxval-minval)
    H = Q[:,0:193] #halo inputs; no baryonic quantities are included here
    M = Q[:,195:201] #Galaxy Mass

    print H.mean(), H.max(), H.min()
    print M.mean(), M.max(), M.min()

    training_size = .6
    valid_size = .5

    H_train, H_valid_test, M_train, M_valid_test = cross_validation.train_test_split(H, M, train_size=training_size, random_state=23) #the random state is chosen for consistency across different runs
    #Split the valid-test split into a validation set and a test set.
    H_valid, H_test, M_valid, M_test = cross_validation.train_test_split(H_valid_test, M_valid_test, train_size=valid_size, random_state=25) #the random state is chosen for consistency across different runs

    #Other set. won't worry about it for now.

    '''
    HB = np.c_[H, Q[:,193:195], M[:,3], M[:,4]] #halo inputs with cooling radius and hot gas from the last two snapshots
    C = np.c_[M[:,1]] #just the cold gas mass

    N_train, N_test, C_train, C_test = cross_validation.train_test_split(HB, C, train_size=training_size, random_state=23)
    '''


    pickle.dump((H_train, M_train), gzip.open('milliTrain.pickle.gz','wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump((H_valid, M_valid), gzip.open('milliValid.pickle.gz','wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump((H_test, M_test), gzip.open('milliTest.pickle.gz','wb'), pickle.HIGHEST_PROTOCOL)