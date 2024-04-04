# import other necessary libaries
import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
from utils import create_line, create_mask

# load the input image
image = cv2.imread('/Users/lcsanchez/Desktop/COMP425/hw2/hough/road.jpg')

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# run Canny edge detector to find edge points
edges = feature.canny(gray)

# display the original image and the edges
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.axis('off')

plt.show()


# get image dimensions
H, W = edges.shape

# create a binary mask for the ROI
mask = create_mask(H, W)

# print the mask
print("Mask:")
print(mask)


# visualize the mask
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.show()

# extract edge points in ROI by multiplying edge map with the mask
edges = np.uint8(edges)
mask = np.uint8(mask)
edges_roi = cv2.bitwise_and(edges, edges, mask=mask)

plt.imshow(edges_roi, cmap='gray')
plt.title('Edges within ROI')
plt.show()

# perform Hough transform
diag_len = int(np.sqrt(edges.shape[0]**2 + edges.shape[1]**2))
rhos = np.arange(-diag_len, diag_len + 1)
thetas = np.deg2rad(np.arange(-90, 90))
cos_thetas = np.cos(thetas)
sin_thetas = np.sin(thetas)

accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
edge_pixels = np.argwhere(edges)
for y, x in edge_pixels:
    for theta_idx, (cos_theta, sin_theta) in enumerate(zip(cos_thetas, sin_thetas)):
        rho = int(x * cos_theta + y * sin_theta)
        rho_idx = np.argmin(np.abs(rhos - rho))
        accumulator[rho_idx, theta_idx] += 1

# find the right lane by finding the peak in hough space
max_rho_idx, max_theta_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
major_lane = create_line(rhos[max_rho_idx], thetas[max_theta_idx], edges_roi)

# zero out the values in accumulator around the neighborhood of the peak
neighborhood_size = 500
accumulator[max(max_rho_idx - neighborhood_size+1, 0):min(max_rho_idx + neighborhood_size+1, accumulator.shape[0]), max(max_theta_idx - neighborhood_size, 0):min(max_theta_idx + neighborhood_size, accumulator.shape[1])] = 0

# find the left lane by finding the peak in hough space
max_rho_idx_2, max_theta_idx_2 = np.unravel_index(np.argmax(accumulator), accumulator.shape)
orange_lane = create_line(rhos[max_rho_idx_2], thetas[max_theta_idx_2], edges_roi)

# plot the results
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.plot(major_lane[0], major_lane[1], color='b', linewidth=4)
plt.plot(orange_lane[0], orange_lane[1], color='orange', linewidth=4)

plt.axis('off')
plt.show()
