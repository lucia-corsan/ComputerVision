import numpy as np
import pickle
from utils import computeHistogram
from scipy.spatial.distance import cdist

with open("model.pkl", "rb") as infile:
    model = pickle.load(infile)
infile.close()

textons = model['textons']
hist_train = model['hist_train']
F = model['F']
K = hist_train.shape[1]

N_img = 7

test_list = []
for i in range(N_img):
    test_list.append('test%d.jpg' % (i+1))

# create histogram of test images
hist_test = np.empty([N_img, K])
for i in range(N_img):
    h = computeHistogram(test_list[i], F, textons)
    hist_test[i,:] = h

D = cdist(hist_test, hist_train)
pred = np.argmin(D, axis = 1)
for i in range(N_img):
    print('For test%d.jpg, the closest training image is train%d.jpg' % (i+1, pred[i]+1))
