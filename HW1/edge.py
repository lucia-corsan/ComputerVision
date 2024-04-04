import numpy as np
import matplotlib.pylab as plt
from skimage import io
from utils import gaussian_kernel, filter2d, partial_x, partial_y

def main():
    # Loading the image
    img = io.imread('iguana.png', as_gray=True)

    # Smooth image with Gaussian kernel
    kernel_size = 5
    sigma = 1.0
    gaussian_filter = gaussian_kernel(kernel_size, sigma)
    smoothed_img = filter2d(img, gaussian_filter)

    # Compute x and y derivates on smoothed image
    img_dx = partial_x(smoothed_img)
    img_dy = partial_y(smoothed_img)

    # Gradient magnitude formula
    gradient_magnitude = np.sqrt(img_dx**2 + img_dy**2)

    # Visualization step
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img_dx, cmap='gray')
    plt.title('X Gradient')

    plt.subplot(1, 3, 2)
    plt.imshow(img_dy, cmap='gray')
    plt.title('Y Gradient')

    plt.subplot(1, 3, 3)
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title('Gradient Magnitude')

    plt.show()
    
if __name__ == "__main__":
    main()

