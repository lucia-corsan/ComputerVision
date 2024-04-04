import matplotlib.pyplot as plt
import numpy as np
from utils import visualize_box, visualize_match
from homography import computeH_ransac, compositeH, matchPics
from skimage import io
from skimage.transform import resize
from skimage.util import img_as_float
from skimage.color import rgb2gray

cv_desk = img_as_float(io.imread('cv_desk.jpg'))
cv_desk_gray = rgb2gray(cv_desk)
cv_cover = img_as_float(io.imread('cv_cover.jpg'))
hp_cover = img_as_float(io.imread('hp_cover.jpg'))

cv_cover_rotated = np.rot90(cv_cover, k=1)

# finding candidate matching pairs between two images
matches, locs1, locs2 = matchPics(cv_cover, cv_desk)

# visualize raw matching result
visualize_match(cv_cover, cv_desk_gray, locs1, locs2, matches)

# resize hp_cover to have the same size of cv_cover
hp_cover_resize = resize(hp_cover, cv_cover.shape, anti_aliasing = True)

# use RANSAC to estimate homograhy and find inlier matches
bestH, inliers = computeH_ransac(matches, locs1, locs2)

# visualize matching result after RANSAC
visualize_match(cv_cover, cv_desk_gray, locs1, locs2, matches[inliers, :])

# visualize the bounding box in target image
visualize_box(cv_cover, cv_desk, bestH)

# create final composite image
composite_img = compositeH(bestH, hp_cover_resize, cv_desk)

# show the final composite image
plt.imshow(composite_img)
plt.show()
