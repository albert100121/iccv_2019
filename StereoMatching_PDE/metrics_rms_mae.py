from PIL import Image
import math
import os
import numpy as np
from file_utilities import list_directories, write_report
import imutils
import cv2
from tqdm import tqdm
from utils import *
import imutils
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
# report_name = "MP3D_StereoMatching_rms_mae.csv"
# data_path_gt = "/home/kike/Documents/Dataset/ICCV_dataset/MP3D/test"

report_name = "SF3D_StereoMatching_PDE_rms_mae.csv"
data_path_gt = "/home/kike/Documents/Dataset/ICCV_dataset/SF3D/test"

dir_disp_gt = os.path.join(data_path_gt, "disp_up")
dir_depth_gt = os.path.join(data_path_gt, "depth_up")
dir_rgb_map_up = os.path.join(data_path_gt, "image_up")
dir_rgb_map_down = os.path.join(data_path_gt, "image_down")

list_disp_gt = list_directories(dir_disp_gt)
list_depth_gt = list_directories(dir_depth_gt)
list_rgb_map_up = list_directories(dir_rgb_map_up)
list_rgb_map_down = list_directories(dir_rgb_map_down)

global_rmse_disp = 0
global_mae_disp = 0
global_rmse_depth = 0
global_mae_depth = 0

stereo = StereoMatching()
pde = PDE()

tbar = tqdm(total=len(list_rgb_map_up))
for i in range(len(list_rgb_map_up)):
    disp_gt = np.load(os.path.join(dir_disp_gt, list_disp_gt[i]))

    mask = disp_gt > 67.5
    disp_gt[mask] = 0

    rgb_map_up = cv2.imread(os.path.join(dir_rgb_map_up, list_rgb_map_up[i]))
    rgb_map_down = cv2.imread(os.path.join(dir_rgb_map_down, list_rgb_map_down[i]))

    imgL = imutils.rotate_bound(rgb_map_up, 270)
    imgR = imutils.rotate_bound(rgb_map_down, 270)
    disp_est_ = stereo.run(imgL, imgR) * 180 / 512

    pde.setting_images(left_image=imgL, right_image=imgR)
    disparity, disparity_before = pde.run(disp_est_)

    disp_est = imutils.rotate_bound(disparity, 90)

    mask1 = disp_gt > 0
    mask2 = disp_est > 0
    disp_est[~mask2] = 0
    disp_gt[~mask1] = 0

    mask = np.logical_or(mask1, mask2)
    rmse_aux_disp = rmse(disp_gt, disp_est, mask)
    mae_aux_disp = mae(disp_gt, disp_est, mask)

    depth_gt = np.load(os.path.join(dir_depth_gt, list_depth_gt[i]))#[:, :, 0]
    depth_est = disp2depth(0.2, disp_est)

    mask1 = np.logical_and(depth_gt > 0, depth_gt < 10)
    mask2 = np.logical_and(depth_est > 0, depth_est < 10)
    depth_est[~mask2] = 0
    depth_gt[~mask1] = 0

    mask = np.logical_and(mask1, mask2)
    mask[0:26, :] = 0
    mask[486:512, :] = 0

    depth_gt[~mask] = 0
    depth_est[~mask] = 0

    rmse_aux_depth = rmse(depth_gt, depth_est, mask)
    mae_aux_depth = mae(depth_gt, depth_est, mask)

    global_rmse_depth += rmse_aux_depth
    global_mae_depth += mae_aux_depth

    global_rmse_disp += rmse_aux_disp
    global_mae_disp += mae_aux_disp

    tbar.update(1)

    line = [i, list_disp_gt[i], rmse_aux_disp, mae_aux_disp, rmse_aux_depth, mae_aux_depth]
    write_report(report_name, line)

tbar.close()

total_rmse_disp = global_rmse_disp / len(list_disp_gt)
total_mae_disp = global_mae_disp / len(list_disp_gt)

total_rmse_depth = global_rmse_depth / len(list_disp_gt)
total_mae_depth = global_mae_depth / len(list_disp_gt)

write_report(report_name, [report_name])
write_report(report_name, ["disp RMSE:", total_rmse_disp])
write_report(report_name, ["disp MAE: ", total_mae_disp])
write_report(report_name, ["depth RMSE: ", total_rmse_depth])
write_report(report_name, ["depth MAE: ", total_mae_depth])
