
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

def negative_log_liklihood(y, y_pred):
    return np.sum((y_pred-y)**2, axis = 0)

SST = negative_log_liklihood(test_labels, test_labels.mean(axis = 0))
SSR = negative_log_liklihood(test_labels, pred_mat)
#print SST
#print SSR
'''
#Calculate R^2
means = test_labels.mean(axis = 0)

SStot = np.zeros(shape = (test_labels.shape[1],))
SSres = np.zeros(shape = (test_labels.shape[1],))

for i in xrange(pred_mat.shape[0]):

    pred = pred_mat[i, :]
    #print 'Prediction: ',pred[0]
    #print 'Actual: ', test_labels[i, 0]
    #print pred, test_labels[i, :]
    #print (pred-test_labels[i, :])**2
    SSres += (test_labels[i, :]-pred)**2
    SStot += (test_labels[i,:] - means)**2
    if i < 10:
        print 'Pred:\t%.4f\nAct:\t%.4f\n'%(pred[0],test_labels[i,0])

print SStot#This calculation is wrong!
print SSres

print (SStot/pred_mat.shape[0]).mean()
print (SSres/pred_mat.shape[0]).mean()
'''
R2 = 1 - SSR/SST
for i in xrange(test_labels.shape[1]):
    print 'R^2 %d is '%(i+1),R2[i]

from matplotlib import pyplot as plt

for idx in xrange(6):
    plt.subplot(211)
    plt.scatter(test_data[:,idx],pred_mat[:, idx])
    x1, x2, y1, y2 = plt.axis()
    plt.title('Prediction')
    plt.subplot(212)
    plt.scatter(test_data[:,idx], test_labels[:,idx])
    plt.title('Actual')
    x1, x2, y3, y4 = plt.axis()
    plt.axis([x1,x2,min(y1, y3), max(y2,y4)])
    plt.subplot(211)
    plt.axis([x1,x2,min(y1, y3), max(y2,y4)])
    plt.show()
    plt.title('Relationship')
    plt.plot(np.linspace(test_labels[:,idx].min(), test_labels[:,idx].max()), np.linspace(test_labels[:,idx].min(), test_labels[:,idx].max()), 'r--')
    plt.scatter(test_labels[:,idx],pred_mat[:, idx])
    plt.show()

