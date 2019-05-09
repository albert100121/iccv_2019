import os
import numpy as np
import cv2
from utils import *

gt_dir = "/home/kike/Documents/Dataset/ICCV_dataset/MP3D/test"
est_dir = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/Whole_estimations/MP3D"

file_name = "17DRP5sb8fy_0_7_"

output_dir = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/GT_visualizations/MP3D/extraScene"

depth_gt = np.load(gt_dir + "/depth_up/" + file_name + "depth.npy")[:, :, 0]
max_depth = depth_gt.max() * 0.8
rgb_map = cv2.imread(os.path.join(gt_dir, "image_up/" + file_name + "color.png"))

# Estimations maps
disp_asw, depth_asw = read_tiff_file(est_dir + "/ASW/disp_up_est/" + file_name + "color.png_pp.tif")
depth_GCNet = np.load(est_dir + "/GCNet/depth_up_est/" + file_name + "color.npy")
depth_PSMNet = np.load(est_dir + "/PSM_ori/depth_up_est/" + file_name + "color.npy")
depth_vCV = np.load(est_dir + "/best_model1_vCV/depth_up_est/" + file_name + "color.npy")
depth_LcV = np.load(est_dir + "/best_model2_LCV/depth_up_est/" + file_name + "color.npy")


depth_gt = standarizing_image(depth_gt, max_depth)
depth_asw = standarizing_image(depth_asw, max_depth)
depth_GCNet = standarizing_image(depth_GCNet, max_depth)
depth_PSMNet = standarizing_image(depth_PSMNet, max_depth)
depth_vCV = standarizing_image(depth_vCV, max_depth)
depth_LcV = standarizing_image(depth_LcV, max_depth)

# plot_image(0, depth_gt, "Depth GT", cmap="jet", filename=os.path.join(output_dir, "depth_GT.png"))
# plt.show()

depth_maps = [depth_asw, depth_GCNet, depth_PSMNet, depth_LcV]
depth_maps_st = ['depth_asw', 'depth_GCNet', 'depth_PSMNet', 'depth_LcV']

for i, depth_est in enumerate(depth_maps):
    error = abs(depth_gt - depth_est)
    mask = (error > 0.05) & (error <= 0.2)
    error_rgb1 = image_masking(error, [0, 255, 0], mask).astype(np.uint8)
    mask = (error > 0.2) & (error <= 0.5)
    error_rgb2 = image_masking(error, [255, 255, 0], mask).astype(np.uint8)
    mask = error > 0.5
    error_rgb3 = image_masking(error, [255, 0, 0], mask).astype(np.uint8)

    error_blend = cv2.addWeighted(error_rgb1, 0.5,
                                  error_rgb2, 1, 0.0)

    error_blend = cv2.addWeighted(error_blend, 1,
                                  error_rgb3, 0.5, 0.0)

    error_blend = cv2.addWeighted(cv2.cvtColor(rgb_map, cv2.COLOR_BGR2RGB), 1,
                                  error_blend, 1, 0.0)

    # error_blend  = cv2.cvtColor(error_blend, cv2.COLOR_BGR2GRAY)

    plot_image(i, error_blend, "{} Disparity Error G:[>0.05]  Y:[>0.2] R:[>0.5]".format(depth_maps_st[i]), False,
               filename=os.path.join(output_dir, "{}_{}_error.png".format(file_name, depth_maps_st[i])))
plt.show()
