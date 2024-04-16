from skimage import io
from skimage.util import img_as_float
from skimage.color import rgb2gray
import numpy as np
from scipy.ndimage import correlate
import sklearn.cluster

def computeHistogram(img_file, F, textons):
    """
    :param img_file: Image
    :param F: Number of filters
    :param textons: List of textons
    :return: Histogram representing the image
    """
    # Load the image
    img = io.imread(img_file)

    # I check if the image is grayscale; if it isn't, I convert it
    if len(img.shape) > 2:
        img = rgb2gray(img)

    # I convert the img to float with scikit-image
    img = img_as_float(img)

    # Initializing 'filter_responses' with the height and width of the image
    # and the depth F (number of filters, 48).
    filter_responses = np.zeros((img.shape[0], img.shape[1], F.shape[2]))

    # I apply each filter to the image using correlate from scipy.ndimage
    for i in range(F.shape[2]):
        filter_responses[..., i] = correlate(img, F[..., i], mode='reflect')

    # Reshaping into a 2D array (each row is the response of a pixel to all filters)
    filter_responses_2d = filter_responses.reshape(-1, F.shape[2])

    # Predicting the clusters for each pixel
    labels = textons.predict(filter_responses_2d)

    # Computing normalized histogram with 'bin edges' = # of clusters
    hist, _ = np.histogram(labels, bins=np.arange(textons.n_clusters + 1), density=True)

    return hist

def createTextons(F, file_list, K):
    """
    :param F: Filter bank
    :param file_list: A list of filenames corresponding to the training images
    :param K: Number of clusters that we want to pursue
    :return: List of textons
    """
    # Collect filter responses from all training images
    filter_responses = []
    for file in file_list:
        img = io.imread(file)
        # I check if the image is grayscale; if it isn't, I convert it
        if len(img.shape) > 2:
            img = rgb2gray(img)
        # I convert the img to float with scikit-image
        img = img_as_float(img)
        # Initializing 'filter_responses' list and flatten it
        response = np.zeros((img.shape[0], img.shape[1], F.shape[2]))
        for i in range(F.shape[2]):
            response[..., i] = correlate(img, F[..., i])
        filter_responses.append(response.reshape(-1, F.shape[2]))

    # Selecting 100 sample pixels per image at random for clustering
    # (7*100-dimension vectors as mentioned in the instructions)
    sampled_responses = np.vstack(filter_responses)
    np.random.shuffle(sampled_responses)
    sampled_responses = sampled_responses[:100 * len(file_list)]

    # I run K-means (from scikit-learn) to cluster the sampled filter responses
    # into K clusters
    textons = sklearn.cluster.KMeans(n_clusters=K, n_init=5)
    textons.fit(sampled_responses)

    return textons
