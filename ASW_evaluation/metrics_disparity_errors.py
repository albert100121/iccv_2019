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
report_name = "MP3D_ASW_disp_errors.csv"
data_path_gt = "/home/kike/Documents/Dataset/ICCV_dataset/MP3D/test"
data_path_est = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/Whole_estimations/MP3D/ASW"

dir_disp_gt = os.path.join(data_path_gt, "disp_up")
dir_disp_est = os.path.join(data_path_est, "disp_up_est")

dir_depth_gt = os.path.join(data_path_gt, "depth_up")
dir_rgb_map = os.path.join(data_path_gt, "image_up")

list_disp_gt = list_directories(dir_disp_gt)
list_disp_est = list_directories(dir_disp_est, key="_pp")
list_depth_gt = list_directories(dir_depth_gt)
list_rgb_map = list_directories(dir_rgb_map)

global_disp_range_1 = 0
global_disp_range_2 = 0
global_disp_range_3 = 0

report = []

tbar = tqdm(total=len(list_disp_est))
for i in range(len(list_disp_est)):
    disp_gt = np.load(os.path.join(dir_disp_gt, list_disp_gt[i]))

    mask_range = disp_gt > 67.5
    disp_gt[mask_range] = 0

    im = Image.open(os.path.join(dir_disp_est, list_disp_est[i]))
    imarray = np.array(im) * (-1)
    disp_est = imutils.rotate_bound(imarray, 90) * 180 / 512

    mask = disp_gt > 0
    disp_est[(mask * (-1) + 1).astype(np.bool)] = 0
    disp_gt[(mask * (-1) + 1).astype(np.bool)] = 0

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

total_dis_error_range_1 = global_disp_range_1 / len(list_disp_est)
total_dis_error_range_2 = global_disp_range_2 / len(list_disp_est)
total_dis_error_range_3 = global_disp_range_3 / len(list_disp_est)

write_report(report_name, [data_path_est])
write_report(report_name, ["disp >0.1:", total_dis_error_range_1])
write_report(report_name, ["disp >0.5: ", total_dis_error_range_2])
write_report(report_name, ["disp >1: ", total_dis_error_range_3])
