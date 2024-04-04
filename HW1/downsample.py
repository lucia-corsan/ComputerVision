import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from utils import gaussian_kernel, filter2d

def main():

    # Loading the image
    im = imread('paint.jpg').astype('float')
    im = im / 255

    # number of levels for downsampling
    N_levels = 5

    # make a copy of the original image
    im_subsample = im.copy()

    # naive subsampling, visualize the results on the 1st row
    for i in range(N_levels):
        # subsample image
        im_subsample = im_subsample[::2, ::2, :]
        plt.subplot(2, N_levels, i + 1)
        plt.imshow(im_subsample)
        plt.axis('off')

        
    # subsampling without aliasing, visualize results on 2nd row
    for i in range(N_levels):
        # Split the image into RGB channels
        r_channel = im[:, :, 0]
        g_channel = im[:, :, 1]
        b_channel = im[:, :, 2]

        # Apply the filter to each channel
        r_filtered = filter2d(r_channel, gaussian_kernel())
        g_filtered = filter2d(g_channel, gaussian_kernel())
        b_filtered = filter2d(b_channel, gaussian_kernel())

        # Combine the filtered channels into a single image
        im = np.stack([r_filtered, g_filtered, b_filtered], axis=-1)

        # Subsampling the image
        im = im[::2, ::2, :]

        # Display the filtered image
        plt.subplot(2, N_levels, N_levels + i + 1)
        plt.imshow(im)
        plt.axis('off')

    plt.show()  



if __name__ == "__main__":
    main()
