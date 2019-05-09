import os
import numpy as np
from file_utilities import list_directories
import cv2
import matplotlib.pyplot as plt
from utils import *
from vispy_utilities import *
from CameraModels.Sphere import Sphere
import vispy

data_path_gt = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/Visualization/reasonable_image_SF3D"
data_path_est = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/Visualization/evaluation/Best_model2_learnCV_SF3D_withpretrain/eval"

dir_depth_gt = os.path.join(data_path_gt, "depth_up_gt")
dir_depth_est = os.path.join(data_path_est, "depth_up_est")
dir_disp_gt = os.path.join(data_path_gt, "disp_up_gt")
dir_disp_est = os.path.join(data_path_est, "disp_up_est")
dir_rgb_map = os.path.join(data_path_gt, "image_up")

list_depth_gt = list_directories(dir_depth_gt)
list_depth_est = list_directories(dir_depth_est)
list_disp_gt = list_directories(dir_disp_gt)
list_disp_est = list_directories(dir_disp_est)
list_rgb_maps = list_directories(dir_rgb_map)

i = 2
depth_gt = np.load(os.path.join(dir_depth_gt, list_depth_gt[i]))#[:, :, 0]
depth_est = np.load(os.path.join(dir_depth_est, list_depth_est[i]))

disp_gt = np.load(os.path.join(dir_disp_gt, list_disp_gt[i]))
disp_est = np.load(os.path.join(dir_disp_est, list_disp_est[i]))

mask = disp_gt > 0
disp_est[(mask * (-1) + 1).astype(np.bool)] = 0
disp_gt[(mask * (-1) + 1).astype(np.bool)] = 0

rgb_map = cv2.imread(os.path.join(dir_rgb_map, list_rgb_maps[i]))

error = abs(disp_gt - disp_est)
mask = (error > 0.1) & (error <= 0.5)
error_rgb1 = image_masking(error, [0, 255, 0], mask).astype(np.uint8)
mask = (error > 0.5) & (error <= 1)
error_rgb2 = image_masking(error, [255, 255, 0], mask).astype(np.uint8)
mask = error > 1
error_rgb3 = image_masking(error, [255, 0, 0], mask).astype(np.uint8)

error_blend = cv2.addWeighted(error_rgb1, 0.5,
                              error_rgb2, 1, 0.0)


error_blend = cv2.addWeighted(error_blend, 1,
                              error_rgb3, 1, 0.0)


error_blend = cv2.addWeighted(cv2.cvtColor(rgb_map, cv2.COLOR_BGR2RGB), 0.2,
                              error_blend, 1, 0.0)


plot_image(1, error_blend, "Disparity Error G:[>0.1]  Y:[>0.5] R:[>1]", False,
           filename=os.path.join(data_path_est, "dispError.png"))

plt.show()