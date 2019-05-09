import os
import numpy as np
import cv2
from utils import *

gt_dir = "/home/kike/Documents/Dataset/ICCV_dataset/SF3D/test"
est_dir = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/Whole_estimations/SF3D"

file_name = "Area_6_conferenceRoom_1_8"

output_dir = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/GT_visualizations/SF3D/extraScene"

disp_gt = np.load(gt_dir + "/disp_up/" + file_name + ".npy")
rgb_map = cv2.imread(os.path.join(gt_dir, "image_up/" + file_name + ".png"))

# Estimations maps
disp_asw, depth_asw = read_tiff_file(est_dir + "/ASW/disp_up_est/" + file_name + ".png_pp.tif")
disp_GCNet = np.load(est_dir + "/GCNet/disp_up_est/" + file_name + ".npy")
disp_PSMNet = np.load(est_dir + "/PSM_ori/disp_up_est/" + file_name + ".npy")
disp_vCV = np.load(est_dir + "/best_model1_vCV/disp_up_est/" + file_name + ".npy")
disp_LcV = np.load(est_dir + "/best_model2_LCV/new/disp_up_est/" + file_name + ".npy")

disp_maps = [disp_GCNet, disp_PSMNet, disp_LcV]
disp_maps_st = ['disp_GCNet', 'disp_PSMNet', 'disp_LcV']

for i, disp_est in enumerate(disp_maps):
    mask = disp_gt > 0
    disp_est[(mask * (-1) + 1).astype(np.bool)] = 0
    disp_gt[(mask * (-1) + 1).astype(np.bool)] = 0

    error = abs(disp_gt - disp_est)
    mask = (error > 0.1) & (error <= 0.5)
    error_rgb1 = image_masking(error, [0, 255, 0], mask).astype(np.uint8)
    mask = (error > 0.5) & (error <= 1)
    error_rgb2 = image_masking(error, [255, 255, 0], mask).astype(np.uint8)
    mask = error > 1
    error_rgb3 = image_masking(error, [255, 0, 0], mask).astype(np.uint8)

    error_blend = cv2.addWeighted(error_rgb1, 1,
                                  error_rgb2, 1, 0.0)

    error_blend = cv2.addWeighted(error_blend, 1,
                                  error_rgb3, 1, 0.0)

    error_blend = cv2.addWeighted(cv2.cvtColor(rgb_map, cv2.COLOR_BGR2RGB), 0.5,
                                  error_blend, 1, 0.0)

    plot_image(i, error_blend, "{} Disparity Error G:[>0.1]  Y:[>0.5] R:[>1]".format(disp_maps_st[i]), False,
               filename=os.path.join(output_dir, "{}_{}_error.png".format(file_name, disp_maps_st[i])))

plt.show()
