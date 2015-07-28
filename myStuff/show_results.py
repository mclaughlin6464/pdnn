
import numpy as np
import sys
import os
import cPickle, gzip
#TODO Not sure what to do with this one, because there isn't a right-wrong to mine. Perhaps average error?
pred_file = sys.argv[1]

if '.gz' in pred_file:
    pred_mat = cPickle.load(gzip.open(pred_file, 'rb'))
else:
    pred_mat = cPickle.load(open(pred_file, 'rb'))

# load the testing set to get the labels
test_data, test_labels = cPickle.load(gzip.open('milliTest.pickle.gz', 'rb'))

#Calculate R^2
means = test_labels.mean(axis = 0)
SStot = np.sum(test_labels-means, axis = 0)**2

print SStot
SSres = np.zeros(shape = (test_labels.shape[1],))
for i in xrange(pred_mat.shape[0]):
    pred = pred_mat[i, :]
    #print 'Prediction: ',pred[0]
    #print 'Actual: ', test_labels[i, 0]
    SSres += (pred-test_labels[i, :])**2

print SSres

R2 = 1 - SSres/SStot
for i in xrange(test_labels.shape[1]):
    print 'R^2 %d is '%(i+1),R2[i]

from matplotlib import pyplot as plt

plt.plot(test_labels[:,0],pred_mat[:, 0])
plt.show()
plt.plot(test_data[:,0])

plt.show()

