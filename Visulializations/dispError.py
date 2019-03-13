import os
import numpy as np
from file_utilities import list_directories
import cv2
import matplotlib.pyplot as plt
from utils import *
from vispy_utilities import *
from CameraModels.Sphere import Sphere
import vispy

data_path = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/Best_model2_learnCV_MP3D/eval"

dir_depth_gt = os.path.join(data_path, "depth_up_gt")
dir_depth_est = os.path.join(data_path, "depth_up_est")
dir_disp_gt = os.path.join(data_path, "disp_up_gt")
dir_disp_est = os.path.join(data_path, "disp_up_est")
dir_rgb_map = os.path.join(data_path, "image_up")

list_depth_gt = list_directories(dir_depth_gt)
list_depth_est = list_directories(dir_depth_est)
list_disp_gt = list_directories(dir_disp_gt)
list_disp_est = list_directories(dir_disp_est)
list_rgb_maps = list_directories(dir_rgb_map)

i = 0
depth_gt = np.load(os.path.join(dir_depth_gt, list_depth_gt[i]))[:, :, 0]
depth_est = np.load(os.path.join(dir_depth_est, list_depth_est[i]))

disp_gt = np.load(os.path.join(dir_disp_gt, list_disp_gt[i]))
disp_est = np.load(os.path.join(dir_disp_est, list_disp_est[i]))

rgb_map = cv2.imread(os.path.join(dir_rgb_map, list_rgb_maps[i]))

error = abs(disp_gt - disp_est)
mask = (error > 0.1) & (error <= 0.5)
error_rgb1 = image_masking(error, [153, 255, 51], mask).astype(np.uint8)
mask = (error > 0.5) & (error <= 1)
error_rgb2 = image_masking(error, [255, 255, 0], mask).astype(np.uint8)
mask = error > 1
error_rgb3 = image_masking(error, [255, 0, 0], mask).astype(np.uint8)

error_blend = cv2.addWeighted(error_rgb1, 0.3,
                              error_rgb2, 1, 0.0)


error_blend = cv2.addWeighted(error_blend, 1,
                              error_rgb3, 1, 0.0)


error_blend = cv2.addWeighted(cv2.cvtColor(rgb_map, cv2.COLOR_BGR2RGB), 0.5,
                              error_blend, 0.8, 0.0)



plt.figure(2)
plt.title("Disparity Error G:[>0.1]  Y:[>0.5] R:[>1]")
plt.imshow(error_blend)
plt.savefig(os.path.join(data_path, "dispError.png"))
plt.show()
