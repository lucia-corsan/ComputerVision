from skimage.feature import match_descriptors, SIFT
from skimage.color import rgb2gray, rgba2rgb, gray2rgb
from skimage import io
import numpy as np
from skimage.transform import warp


def matchPics(I1, I2):
    # Given two images I1 and I2, perform SIFT matching to find candidate match pairs

    # Convert images to grayscale if necessary
    if len(I1.shape) == 3:
        I1_gray = rgb2gray(I1)
    else:
        I1_gray = I1.copy()

    if len(I2.shape) == 3:
        I2_gray = rgb2gray(I2)
    else:
        I2_gray = I2.copy()

    # Initialize SIFT detector
    sift = SIFT()

    # Detect the keypoints
    sift.detect_and_extract(I1_gray)
    kp1 = sift.keypoints
    # Compute the descriptors
    desc1 = sift.descriptors
    # Get keypoint locations
    locs1 =  sift.positions

    # Same for the second image
    sift.detect_and_extract(I2_gray)
    kp2 = sift.keypoints
    desc2 = sift.descriptors
    locs2 = sift.positions

    # Match descriptors between the two images
    matches = match_descriptors(desc1, desc2, max_ratio=0.6, cross_check=True)

    return matches, locs1, locs2

def computeH_ransac(matches, locs1, locs2):

    ### You should implement this function using Numpy only
    locs1 = np.flip(locs1, axis=1)
    locs2 = np.flip(locs2, axis=1)

    num_iterations = 1500
    threshold = 0.7
    best_inliers = np.array([])

    for _ in range(num_iterations):
        # Randomly select 4 matches
        idx = np.random.choice(len(matches), 4, replace=False)
        selected_matches = [matches[i] for i in idx]
        selected_locs1 = locs1[[m[0] for m in selected_matches]]
        selected_locs2 = locs2[[m[1] for m in selected_matches]]

        # Compute the homography
        H = computeHomography(selected_locs1, selected_locs2)

        # Calculate inliers
        inliers = np.array([idx for idx, (i, j) in enumerate(matches) if dist(locs1[i], locs2[j], H) < threshold])

        # Update bestH and best_inliers if we found a better model
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    bestH = computeHomography(locs1[matches[best_inliers, 0]], locs2[matches[best_inliers, 1]])

    return bestH, best_inliers


def compositeH(H, template, img):
    # Create a compositie image after warping the template image on top
    # of the image using homography

    # Create mask of same size as template

    # Warp mask by appropriate homography
    mask = np.ones_like(template)

    # Warp template by appropriate homography
    warped_mask = warp(mask, np.linalg.inv(H), output_shape=img.shape[:2])

    if len(template.shape) == 2:
        template = gray2rgb(template)

    warped_template = warp(template, np.linalg.inv(H), output_shape=img.shape)

    # Use mask to combine the warped template and the image
    composite_img = img.copy()
    composite_img[warped_mask > 0] = warped_template[warped_mask > 0]

    return composite_img

def computeHomography(locs1, locs2):
    """Solves for the homography given corresponding points in two images.

    Args:
        locs1 (np.ndarray): Array of shape (N, 2) representing N (x, y) coordinates in the first image.
        locs2 (np.ndarray): Array of shape (N, 2) representing N (x, y) coordinates in the second image.

    Returns:
        np.ndarray: The computed homography matrix.
    """
    A = []
    for pt1, pt2 in zip(locs1, locs2):
        x1, y1 = pt1
        x2, y2 = pt2
        A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, -y2 * x1, y2 * y1, y2])
    A = np.array(A)

    # Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(A)

    # V has shape (9, 9) for any number of input pairs. V[-1] is the eigenvector
    # of (A^T)A with the smallest eigenvalue. Reshape into 3x3 matrix.
    H = np.reshape(V[-1], (3, 3))

    # Normalization
    H = H / H[-1,-1]
    return H


def dist(pt1, pt2, H):
    """Returns the geometric distance between two corresponding points given the homography H.

    Args:
        pt1 (np.ndarray): Array of shape (2,) representing (x, y) coordinates of a point in the first image.
        pt2 (np.ndarray): Array of shape (2,) representing (x, y) coordinates of the corresponding point in the second image.
        H (np.ndarray): The homography matrix.

    Returns:
        float: The geometric distance between the points.
    """
    # Points in homogeneous coordinates
    p1 = np.array([pt1[0], pt1[1], 1])
    p2 = np.array([pt2[0], pt2[1], 1])

    # Estimate transformed point
    p2_estimate = np.dot(H, p1)
    p2_estimate = (1 / p2_estimate[2]) * p2_estimate

    # Calculate Euclidean distance
    return np.linalg.norm(p2[:2] - p2_estimate[:2])



