import numpy as np
from scipy.ndimage import gaussian_filter

from utils import filter2d, partial_x, partial_y
from skimage.feature import peak_local_max
from skimage.io import imread
import matplotlib.pyplot as plt

def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    response = None
    
    # Smooth image with Gaussian kernel
    sigma = 1.0
    smoothed_img = gaussian_filter(img, sigma)

    # Now I compute the partial derivatives (gradients)
    Ix = partial_x(smoothed_img)
    Iy = partial_y(smoothed_img)

    # Elements of the tensor M
    Ix2 = Ix**2
    Iy2 = Iy**2
    Ixy = Ix * Iy

    # Sums of the elements in the window
    Sx2 = filter2d(Ix2, np.ones((window_size, window_size)))
    Sy2 = filter2d(Iy2, np.ones((window_size, window_size)))
    Sxy = filter2d(Ixy, np.ones((window_size, window_size)))

    # Harris response
    det_M = Sx2 * Sy2 - Sxy**2
    trace_M = Sx2 + Sy2
    response = det_M - k * trace_M**2

    return response

def main():
    
    # Loading the image
    img = imread('building.jpg', as_gray=True)

    # Window size = 4 considers a small local neighborhood around each pixel to compute the gradient and structure tensor.
    response = harris_corners(img, window_size=4, k=0.04)

    # Threshold on response
    threshold = 0.005 * np.max(response)
    corner_map = response > threshold

    # NMS by finding the peak local maximum
    corners = peak_local_max(response, min_distance=5, threshold_abs=threshold)

    # Visualization
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(response, cmap='gray')
    plt.title('Harris Response')

    plt.subplot(1, 3, 2)
    plt.imshow(corner_map, cmap='gray')
    plt.title('Thresholded Response')

    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap='gray')
    plt.scatter(corners[:, 1], corners[:, 0], marker='x', color='r')
    plt.title('Detected Corners')

    plt.show()


if __name__ == "__main__":
    main()
