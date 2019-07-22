import os
import numpy as np
from file_utilities import list_directories
import cv2
from utils import *

data_path_gt = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/Visualization for kike/reasonable_image_MP3D"
data_path_est = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/Visualization for kike/evaluation/Best_model1_vertCV_MP3D/eval"

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

i = 0
depth_gt = np.load(os.path.join(dir_depth_gt, list_depth_gt[i]))[:, :, 0]
depth_est = np.load(os.path.join(dir_depth_est, list_depth_est[i]))

disp_gt = np.load(os.path.join(dir_disp_gt, list_disp_gt[i]))
disp_est = np.load(os.path.join(dir_disp_est, list_disp_est[i]))

rgb_map = cv2.imread(os.path.join(dir_rgb_map, list_rgb_maps[i]))

image = depth_gt
mask = (image > 0) & (image <= 2)
error_rgb1 = image_masking(image, [255, 0, 0], mask).astype(np.uint8)
mask = (image > 2) & (image <= 5)
error_rgb2 = image_masking(image, [255, 0, 0], mask).astype(np.uint8)
mask = image > 5
error_rgb3 = image_masking(image, [255, 0, 0], mask).astype(np.uint8)

error_blend = cv2.addWeighted(error_rgb1, 0.1,
                              error_rgb2, 0.5, 0.0)


error_blend1 = cv2.addWeighted(error_blend, 1,
                              error_rgb3, 0.8, 0.0)


error_blend = cv2.addWeighted(cv2.cvtColor(rgb_map, cv2.COLOR_BGR2RGB), 1,
                              error_blend1, 1, 0.0)


plot_image(1, error_blend, "depths by range", False,
           filename=os.path.join(data_path_est, "image_by_range.png"))

plot_image(2, error_blend1, "depths by range", True,
           filename=os.path.join(data_path_est, "image_by_range2.png"))
plt.show()