
import numpy
import sys
import os
import cPickle, gzip
import numpy as np


pred_file = sys.argv[1]

if '.gz' in pred_file:
    pred_mat = cPickle.load(gzip.open(pred_file, 'rb'))
else:
    pred_mat = cPickle.load(open(pred_file, 'rb'))

pred_vec = np.zeros((pred_mat.shape[0],))
# load the testing set to get the labels
test_data, test_labels = cPickle.load(gzip.open('test.pickle.gz', 'rb'))
test_labels = test_labels.astype(numpy.int32)

correct_number = 0.0
for i in xrange(pred_mat.shape[0]):
    p = pred_mat[i, :]
    p_sorted = (-p).argsort()
    if p_sorted[0] == test_labels[i]:
        correct_number += 1
    pred_vec[i] = p_sorted[0]

# output the final error rate
error_rate = 100 * (1.0 - correct_number / pred_mat.shape[0])
print 'Error rate is ' + str(error_rate) + ' (%)'

from matplotlib import pyplot as plt

plt.subplot(211)
plt.scatter(test_data[:,0],pred_vec)
x1, x2, y1, y2 = plt.axis()
plt.title('Prediction')
plt.subplot(212)
plt.scatter(test_data[:,0], test_labels)
plt.title('Actual')
x1, x2, y3, y4 = plt.axis()
plt.axis([x1,x2,min(y1, y3), max(y2,y4)])
plt.subplot(211)
plt.axis([x1,x2,min(y1, y3), max(y2,y4)])
plt.show()
plt.title('Relationship')
plt.plot(np.linspace(pred_vec.min(), pred_vec.max()), np.linspace(test_labels.min(), test_labels.max()), 'r-')
plt.scatter(test_labels,pred_vec)
plt.show()
