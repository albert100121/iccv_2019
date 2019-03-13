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


def rmse(im1, im2, mask):
    return math.sqrt(mean_squared_error(im1, im2) * 512 * 1024 / np.count_nonzero(mask))


def mae(im1, im2, mask):
    return mean_absolute_error(im1, im2) * 512 * 1024 / np.count_nonzero(mask)


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
report_name = "PSM_ori_disp_error.csv"
data_path_gt = "/home/kike/Documents/Dataset/ICCV_dataset/MP3D/test"
data_path_est = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/Whole_estimations/MP3D/PSM_ori"

dir_disp_gt = os.path.join(data_path_gt, "disp_up")
dir_disp_est = os.path.join(data_path_est, "disp_up_est")

dir_depth_gt = os.path.join(data_path_gt, "depth_up")
dir_depth_est = os.path.join(data_path_est, "depth_up_est")
dir_rgb_map = os.path.join(data_path_gt, "image_up")

list_disp_gt = list_directories(dir_disp_gt)
list_disp_est = list_directories(dir_disp_est)
list_depth_gt = list_directories(dir_depth_gt)
list_depth_est = list_directories(dir_depth_est)
list_rgb_map = list_directories(dir_rgb_map)

global_rmse_disp = 0
global_mae_disp = 0
global_rmse_depth = 0
global_mae_depth = 0

report = []

tbar = tqdm(total=len(list_disp_est))
for i in range(len(list_disp_est)):
    try:
        disp_gt = np.load(os.path.join(dir_disp_gt, list_disp_gt[i]))
    except:
        print("File {} was not loaded".format(list_disp_gt[i]))
        continue

    mask = disp_gt > 67.5
    disp_gt[mask] = 0

    disp_est = np.load(os.path.join(dir_disp_est, list_disp_est[i]))

    mask = disp_gt > 0

    error =
    disp_range_1 +=
    rmse_aux_disp = rmse(disp_gt, disp_est, mask)
    mae_aux_disp = mae(disp_gt, disp_est, mask)

    depth_gt = disp2depth(0.2, disp_gt)
    depth_est = disp2depth(0.2, disp_est)

    global_rmse_disp += rmse_aux_disp
    global_mae_disp += mae_aux_disp

    tbar.update(1)

    line = [i, list_disp_gt[i], rmse_aux_disp, mae_aux_disp, rmse_aux_depth, mae_aux_depth]
    write_report(report_name, line)

tbar.close()

total_rmse_disp = global_rmse_disp / len(list_disp_est)
total_mae_disp = global_mae_disp / len(list_disp_est)

total_rmse_depth = global_rmse_depth / len(list_disp_est)
total_mae_depth = global_mae_depth / len(list_disp_est)

line = [data_path_est, total_rmse_disp, total_mae_disp, total_rmse_depth, total_mae_depth]
write_report(report_name, line)
