from LMFilters import makeLMfilters
import numpy as np
import pickle
from utils import computeHistogram, createTextons

N_img = 7 # number of train/test images
K = 50 # number of clusters

# create and visualize filter banks
F = makeLMfilters()
N_filters = F.shape[2]

train_list = []

for i in range(N_img):
    train_list.append('train%d.jpg' % (i+1))

# create textons
textons = createTextons(F, train_list, K)

# create histogram of training images
hist_train = np.empty([N_img, K])
for i in range(N_img):
    h = computeHistogram(train_list[i], F, textons)
    hist_train[i,:] = h

# save results 
model = {'textons': textons, 'hist_train': hist_train, 'F': F}
with open("model.pkl", "wb") as outfile:
    pickle.dump(model, outfile)
outfile.close()
