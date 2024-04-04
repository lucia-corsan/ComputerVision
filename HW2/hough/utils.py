import numpy as np

def create_mask(H, W):
    # Generate mask for ROI (Region of Interest)
    mask = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            if i > (float(H) / W) * j and i > -(float(H) / W) * j + H:
                mask[i, j] = 1
                mask[H-3:H, :] = 0
    return mask

def create_line(rho, theta, img):
    xs = []
    ys = []
    # Transform a point in Hough space to a line in xy-space.
    a = - (np.cos(theta)/np.sin(theta)) # slope of the line
    b = (rho/np.sin(theta)) # y-intersect of
    for x in range(img.shape[1]):
        y = a * x + b
        if y > img.shape[0] * 0.6 and y < img.shape[0]:
            xs.append(x)
            ys.append(int(round(y)))

    return xs, ys