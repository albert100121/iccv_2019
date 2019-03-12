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
    return math.sqrt(mean_squared_error(im1[mask], im2[mask]) * 512 * 1024 / np.count_nonzero(mask))


def mae(im1, im2, mask):
    return mean_absolute_error(im1[mask], im2[mask]) * 512 * 1024 / np.count_nonzero(mask)


data_path = "/home/kike/Documents/Dataset/ICCV_dataset/MP3D/test"
dir_disp_gt = os.path.join(data_path, "disp_up")
dir_disp_est = os.path.join(data_path, "asw_estimation")
dir_depth_gt = os.path.join(data_path, "depth_up")
dir_rgb_map = os.path.join(data_path, "image_up")

list_disp_gt = list_directories(dir_disp_gt)
list_disp_est = list_directories(dir_disp_est, key="_pp")
list_depth_gt = list_directories(dir_depth_gt)
list_rgb_map = list_directories(dir_rgb_map)

global_rmse_disp = 0
global_mae_disp = 0
global_rmse_depth = 0
global_mae_depth = 0

tbar = tqdm(total=len(list_disp_est))
for i in range(len(list_disp_est)):
    rgb_map = cv2.imread(os.path.join(dir_rgb_map, list_rgb_map[i]))
    disp_gt = np.load(os.path.join(dir_disp_gt, list_disp_gt[i]))

    mask = disp_gt > 192
    disp_gt[mask] = 0

    im = Image.open(os.path.join(dir_disp_est, list_disp_est[i]))
    imarray = np.array(im) * (-1)
    disp_asw = imutils.rotate_bound(imarray, 90) * 180 / 512

    mask = rgb_map > 0
    mask = mask[:, :, 0] * mask[:, :, 1] * mask[:, :, 2]
    rmse_aux_disp = rmse(disp_gt, disp_asw, mask)
    mae_aux_disp = mae(disp_gt, disp_asw, mask)

    depth_gt = disp2depth(0.2, disp_gt)
    depth_asw = disp2depth(0.2, disp_asw)

    mask[0:26, :] = 0
    mask[486:512, :] = 0

    rmse_aux_depth = rmse(depth_gt, depth_asw, mask)
    mae_aux_depth = mae(depth_gt, depth_asw, mask)

    global_rmse_depth += rmse_aux_depth
    global_mae_depth += mae_aux_depth

    global_rmse_disp += rmse_aux_disp
    global_mae_disp += mae_aux_disp

    tbar.update(1)

    line = [list_disp_gt[i], rmse_aux_disp, mae_aux_disp, rmse_aux_depth, mae_aux_depth]
    write_report("report.csv", line)


tbar.close()

total_rmse_disp = global_rmse_disp / len(list_disp_est)
total_mae_disp = global_mae_disp / len(list_disp_est)

total_rmse_depth = global_rmse_depth / len(list_disp_est)
total_mae_depth = global_mae_depth / len(list_disp_est)

line = [data_path, total_rmse_disp, total_mae_disp, total_rmse_depth, total_mae_depth]
write_report("report.csv", line)
