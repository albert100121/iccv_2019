import os
import numpy as np
import cv2
from vispy_utilities import *
from utils import *

gt_dir = "/home/kike/Documents/Dataset/ICCV_dataset/SF3D/test"
est_dir = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/Whole_estimations/SF3D"

file_name = "Area_6_conferenceRoom_1_8"

output_dir = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/GT_visualizations/SF3D/extraScene"

depth_gt = np.load(gt_dir + "/depth_up/" + file_name + ".npy")
disp_gt = np.load(gt_dir + "/disp_up/" + file_name + ".npy")
max_depth = depth_gt.max()

# Estimations maps
disp_asw, depth_asw = read_tiff_file(est_dir + "/ASW/disp_up_est/" + file_name + ".png_pp.tif")

depth_GCNet = np.load(est_dir + "/GCNet/depth_up_est/" + file_name + ".npy")
disp_GCNet = np.load(est_dir + "/GCNet/disp_up_est/" + file_name + ".npy")

depth_PSMNet = np.load(est_dir + "/PSM_ori/depth_up_est/" + file_name + ".npy")
disp_PSMNet = np.load(est_dir + "/PSM_ori/disp_up_est/" + file_name + ".npy")

depth_vCV = np.load(est_dir + "/best_model1_vCV/depth_up_est/" + file_name + ".npy")
disp_vCV = np.load(est_dir + "/best_model1_vCV/disp_up_est/" + file_name + ".npy")

depth_LcV = np.load(est_dir + "/best_model2_LCV/depth_up_est/" + file_name + ".npy")
disp_LcV = np.load(est_dir + "/best_model2_LCV/disp_up_est/" + file_name + ".npy")

# disp_subpixel = np.load(est_dir + "/subpixel/disp_up_est/" + file_name + "color.npy")
# depth_subpixel = np.load(est_dir + "/subpixel/depth_up_est/" + file_name + "color.npy")


depth_gt = standarizing_image(depth_gt, max_depth)
depth_asw = standarizing_image(depth_asw, max_depth)
depth_GCNet = standarizing_image(depth_GCNet, max_depth)
depth_PSMNet = standarizing_image(depth_PSMNet, max_depth)
depth_vCV = standarizing_image(depth_vCV, max_depth)
depth_LcV = standarizing_image(depth_LcV, max_depth)

depth_cmap = "jet"
mask = disp_gt > 20
disp_gt[mask] = 20
vmax = 4
plot_image(1, depth_gt, "Depth GT", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_depth_gt.png".format(file_name)))
# plot_image(3, depth_asw, "Depth ASW", cmap=depth_cmap, filename=os.path.join(output_dir, "{}_depth_ASW.png".format(file_name)))
plot_image(5, depth_PSMNet, "Depth PSMNet", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_depth_PSMNet.png".format(file_name)))
plot_image(7, depth_GCNet, "Depth GCNet", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_depth_GCNet.png".format(file_name)))
plot_image(11, depth_LcV, "Depth LcV", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_depth_LcV.png".format(file_name)))

vmax = 15
plot_image(2, disp_gt, "Disp GT", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_disp_gt.png".format(file_name)))
# plot_image(4, disp_asw, "Disp ASW", cmap=depth_cmap, filename=os.path.join(output_dir, "{}_disp_ASW.png".format(file_name)))
plot_image(6, disp_PSMNet, "Disp PSMNet", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_disp_PSMNet.png".format(file_name)))
plot_image(8, disp_GCNet, "Disp GCNet", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_disp_GCNet.png".format(file_name)))
plot_image(12, disp_LcV, "Disp LcV", vmax=vmax, cmap=depth_cmap, filename=os.path.join(output_dir, "{}_disp_LcV.png".format(file_name)))

#
# for idx, depth_est in enumerate([depth_gt, depth_PSMNet, depth_LcV]):
#     error = abs(depth_gt - depth_est)
#     plot_image(idx, error, "Error depth", True, cmap=depth_cmap, filename=os.path.join(output_dir, "ErrorDepth.png"))

plt.show()
