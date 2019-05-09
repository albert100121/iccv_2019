import os
import numpy as np
import cv2
from vispy_utilities import *
from utils import *

gt_dir = "/home/kike/Documents/Dataset/ICCV_dataset/MP3D/test/fix_dot"
est_dir = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/Whole_estimations/MP3D"

file_name = "17DRP5sb8fy_0_7_"

output_dir = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/GT_visualizations/MP3D/extraScene"

depth_gt = np.load(gt_dir + "/depth_up/" + file_name + "depth.npy")[:, :, 0]
disp_gt = np.load(gt_dir + "/disp_up/" + file_name + "color.npy")
max_depth = depth_gt.max() * 0.8

# Estimations maps
disp_asw, depth_asw = read_tiff_file(est_dir + "/ASW/disp_up_est/" + file_name + "color.png_pp.tif")

depth_GCnet = np.load(est_dir + "/GCNet/depth_up_est/" + file_name + "color.npy")
disp_GCNet = np.load(est_dir + "/GCNet/disp_up_est/" + file_name + "color.npy")

depth_PSMNet = np.load(est_dir + "/PSM_ori/depth_up_est/" + file_name + "color.npy")
disp_PSMNet = np.load(est_dir + "/PSM_ori/disp_up_est/" + file_name + "color.npy")

depth_vCV = np.load(est_dir + "/best_model1_vCV/depth_up_est/" + file_name + "color.npy")
disp_vCV = np.load(est_dir + "/best_model1_vCV/disp_up_est/" + file_name + "color.npy")

depth_LcV = np.load(est_dir + "/best_model2_LCV/depth_up_est/" + file_name + "color.npy")
disp_LcV = np.load(est_dir + "/best_model2_LCV/disp_up_est/" + file_name + "color.npy")

# disp_subpixel = np.load(est_dir + "/subpixel/disp_up_est/" + file_name + "color.npy")
# depth_subpixel = np.load(est_dir + "/subpixel/depth_up_est/" + file_name + "color.npy")


depth_gt = standarizing_image(depth_gt, max_depth)
depth_asw = standarizing_image(depth_asw, max_depth)
depth_GCnet = standarizing_image(depth_GCnet, max_depth)
depth_PSMNet = standarizing_image(depth_PSMNet, max_depth)
depth_vCV = standarizing_image(depth_vCV, max_depth)
depth_LcV = standarizing_image(depth_LcV, max_depth)

depth_cmap = "jet"
vmax = 10
plot_image(1, depth_gt, "Depth GT", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_depth_gt.png".format(file_name)))
plot_image(3, depth_asw, "Depth ASW", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_depth_ASW.png".format(file_name)))
plot_image(5, depth_PSMNet, "Depth PSMNet", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_depth_PSMNet.png".format(file_name)))
plot_image(7, depth_GCnet, "Depth GCNet", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_depth_GCNet.png".format(file_name)))
plot_image(11, depth_LcV, "Depth LcV", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_depth_LcV.png".format(file_name)))

vmax = 20
plot_image(2, disp_gt, "Disp GT", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_disp_gt.png".format(file_name)))
plot_image(4, disp_asw, "Disp ASW", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_disp_ASW.png".format(file_name)))
plot_image(6, disp_PSMNet, "Disp PSMNet", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_disp_PSMNet.png".format(file_name)))
plot_image(8, disp_GCNet, "Disp GCNet", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_disp_GCNet.png".format(file_name)))
plot_image(12, disp_LcV, "Disp LcV", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_disp_LcV.png".format(file_name)))

#
# for idx, depth_est in enumerate([depth_gt, depth_PSMNet, depth_LcV]):
#     error = abs(depth_gt - depth_est)
#     # mask = (error > 2)
#     # error[mask] = 2
#     plot_image(idx, error, "Error depth", True, cmap=depth_cmap, filename=os.path.join(output_dir, "ErrorDepth.png"))

plt.show()
