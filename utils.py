import numpy as np
import matplotlib.pylab as plt

def visualize_box(template, target, H):
    # visualize the detected box of the template in target
    nrow, ncol = template.shape[:2]
    row = np.array([0, 0, nrow -1, nrow - 1, 0]).astype(float)
    col = np.array([0, ncol-1, ncol-1, 0, 0]).astype(float)
    x = H[0,0]*col + H[0,1]*row + H[0,2]
    y = H[1,0]*col + H[1,1]*row + H[1,2]
    w = H[2,0]*col + H[2,1]*row + H[2,2]
    x = x / w; y = y / w
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    plt.imshow(target)
    plt.plot(x, y, 'r-')
    plt.show()    


def visualize_match(template, target, locs1, locs2, matches):
    # assume template has fewer rows. Both template/target are grayscale
    assert(template.ndim == 2)
    assert(target.ndim == 2)
    nrow1, ncol1 = template.shape[:2]
    nrow2 = target.shape[0]
    template = np.pad(template, ((0, nrow2 - nrow1), (0, 0)))
    
    img = np.hstack((template, target))

    plt.imshow(img, cmap='gray') 
    i1 = matches[:, 0]; i2 = matches[:, 1]
    x1 = locs1[i1, 1]; y1 = locs1[i1, 0]
    x2 = locs2[i2, 1] + ncol1; y2 = locs2[i2, 0]
    plt.plot([x1, x2], [y1, y2])
    plt.show()

