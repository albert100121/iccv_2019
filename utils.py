import math
import numpy as np


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