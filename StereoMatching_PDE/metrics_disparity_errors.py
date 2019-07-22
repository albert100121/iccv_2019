from PIL import Image
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import numpy as np
from file_utilities import list_directories, write_report
import imutils
import cv2
from tqdm import tqdm
from utils import *
from CV_stereo_matching.opencv_stereo_matching import StereoMatching
from PDE import PDE

"""
    data_path --> dir of the dataset
    The folders inside have to be the following structure tree
    main dir
        |--- asw_estimation : disparity estimation
        |--- disp_up: GT disparity
        |--- depth_up: GT depth map
        |--- image_up
        |--- imag_down
"""
report_name = "SF3D_StereoMatching_disp_range_PDE.csv"
data_path_gt = "/home/kike/Documents/Dataset/ICCV_dataset/SF3D/test"

# report_name = "SF3D_StereoMatching_disp_range.csv"
# data_path_gt = "/home/kike/Documents/Dataset/ICCV_dataset/SF3D/test"


dir_disp_gt = os.path.join(data_path_gt, "disp_up")
dir_depth_gt = os.path.join(data_path_gt, "depth_up")
dir_rgb_map_up = os.path.join(data_path_gt, "image_up")
dir_rgb_map_down = os.path.join(data_path_gt, "image_down")

list_disp_gt = list_directories(dir_disp_gt)
list_depth_gt = list_directories(dir_depth_gt)
list_rgb_map_up = list_directories(dir_rgb_map_up)
list_rgb_map_down = list_directories(dir_rgb_map_down)

global_disp_range_1 = 0
global_disp_range_2 = 0
global_disp_range_3 = 0

report = []
pde = PDE()
stereo = StereoMatching()
tbar = tqdm(total=len(list_disp_gt))
for i in range(len(list_disp_gt)):
    disp_gt = np.load(os.path.join(dir_disp_gt, list_disp_gt[i]))

    mask_range = disp_gt > 67.5
    disp_gt[mask_range] = 0

    rgb_map_up = cv2.imread(os.path.join(dir_rgb_map_up, list_rgb_map_up[i]))
    rgb_map_down = cv2.imread(os.path.join(dir_rgb_map_down, list_rgb_map_down[i]))

    imgL = imutils.rotate_bound(rgb_map_up, 270)
    imgR = imutils.rotate_bound(rgb_map_down, 270)
    disp_est = stereo.run(imgL, imgR) * 180 / 512

    pde.setting_images(left_image=imgL, right_image=imgR)
    disparity, disparity_before = pde.run(disp_est)

    disp_est = imutils.rotate_bound(disparity, 90)

    mask1 = disp_gt > 0
    mask2 = disp_est > 0
    disp_est[~mask2] = 0
    disp_gt[~mask1] = 0

    disp_error = np.abs(disp_gt - disp_est)

    disp_range_1 = disp_error >= 0.1
    disp_range_2 = disp_error >= 0.5
    disp_range_3 = disp_error >= 1

    global_disp_range_1 += np.count_nonzero(disp_range_1) / (disp_range_1.size)
    global_disp_range_2 += np.count_nonzero(disp_range_2) / (disp_range_2.size)
    global_disp_range_3 += np.count_nonzero(disp_range_3) / (disp_range_3.size)

    tbar.update(1)

    line = [i, list_disp_gt[i], np.count_nonzero(disp_range_1) / (disp_range_1.size),
            np.count_nonzero(disp_range_2) / (disp_range_2.size),
            np.count_nonzero(disp_range_3) / (disp_range_3.size)]
    write_report(report_name, line)

tbar.close()

total_dis_error_range_1 = global_disp_range_1 / len(list_disp_gt)
total_dis_error_range_2 = global_disp_range_2 / len(list_disp_gt)
total_dis_error_range_3 = global_disp_range_3 / len(list_disp_gt)

write_report(report_name, [report_name])
write_report(report_name, ["disp >0.1:", total_dis_error_range_1])
write_report(report_name, ["disp >0.5: ", total_dis_error_range_2])
write_report(report_name, ["disp >1: ", total_dis_error_range_3])
