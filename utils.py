import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import cv2
import scipy.misc

def disp2depth(baseline, disp):
    angle = np.zeros((512, 1024))
    angle2 = np.zeros((512, 1024))
    for i in range(1024):
        for j in range(512):
            theta_T = math.pi - ((j + 0.5) * math.pi / 512)
            angle[j, i] = baseline * math.sin(theta_T)
            angle2[j, i] = baseline * math.cos(theta_T)

    mask = disp > 0

    depth = np.zeros_like(disp).astype(np.float)
    depth[mask] = (angle[mask] / np.tan(disp[mask] / 180 * math.pi)) + angle2[mask]
    return depth


def image_masking(image, color, mask):
    h, w = image.shape
    image_mask = np.zeros((h, w, 3))
    for idx, i in enumerate(color):
        image_mask[:, :, idx] = i * mask

    return image_mask


def plot_image(idx, image, tittle="", bar=False, cmap='RdYlBu', filename=None, vmax=20):
    save = False
    if filename is not None:
        save = True
    plt.figure(idx)
    plt.title(tittle)
    plt.imshow(image, cmap=cmap, vmax=vmax)
    if bar:
        plt.colorbar(orientation='horizontal')
    plt.grid(False)
    plt.axis('off')
    if save:
        plt.savefig(filename)
        # cv2.imwrite(filename, image)
        # scipy.misc.imsave(filename, image)


def rmse(im1, im2, mask):
    return math.sqrt(mean_squared_error(im1, im2) * 512 * 1024 / np.count_nonzero(mask))


def mae(im1, im2, mask):
    return mean_absolute_error(im1, im2) * 512 * 1024 / np.count_nonzero(mask)


def read_tiff_file(path):
    from PIL import Image
    import imutils
    im = Image.open(path)
    imarray = np.array(im) * (-1)
    disp_est = imutils.rotate_bound(imarray, 90) * 180 / 512
    depth_est = disp2depth(0.2, disp_est)
    return disp_est, depth_est


def standarizing_image(image, threshold):
    mask = image > threshold
    image[mask] = threshold

    image[0:26, :] = 0
    image[486:512, :] = 0

    # image = image / threshold
    # image = (image * 255).astype(np.uint8)
    return image


def standarizing_image_disp(image):
    mask = image > 0
    image[(mask * (-1) + 1).astype(np.bool)] = 0
    return image