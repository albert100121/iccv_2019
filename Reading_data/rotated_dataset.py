import os
import numpy as np
from file_utilities import list_directories
import cv2
import imutils
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

data_path = "/home/kike/Documents/Dataset/ICCV_dataset/MP3D/test"
dir_image_up = os.path.join(data_path, "image_up_rotated")
dir_image_down = os.path.join(data_path, "image_down_rotated")

list_image_up = list_directories(dir_image_up)
list_image_down = list_directories(dir_image_down)

assert len(list_image_down) == len(list_image_up)

tbar = tqdm(total=len(list_image_up))
for i in range(len(list_image_up)):
    image_up = cv2.imread(os.path.join(dir_image_up, list_image_up[i]))
    imarray = np.array(image_up)
    rotated = imutils.rotate_bound(imarray, -90)
    cv2.imwrite(os.path.join(dir_image_up, list_image_up[i]), rotated)
    image_down = cv2.imread(os.path.join(dir_image_down, list_image_down[i]))
    imarray = np.array(image_down)
    rotated = imutils.rotate_bound(imarray, -90)
    cv2.imwrite(os.path.join(dir_image_down, list_image_down[i]), rotated)
    tbar.update(1)
tbar.close()