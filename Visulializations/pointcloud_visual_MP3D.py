import os, sys

sys.path.append(os.path.join(os.environ['HOME'], "Documents/PycharmProjects/Utilities"))
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from vispy_utilities import *
from CameraModels.Sphere import Sphere
import vispy
import numpy as np
import cv2

# from utils import *


data_path_gt = "/home/kike/Documents/Dataset/ICCV_dataset/MP3D/test/fix_dot"
file = "17DRP5sb8fy_0_0_"

# data_path_est = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/Whole_estimations/MP3D/best_model2_LCV"
data_path_est = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/Whole_estimations/MP3D/best_model2_LCV"
# data_path_est = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/Whole_estimations/MP3D/PSM_ori"

# data_path_est ="/home/kike/Documents/Dataset/ICCV_dataset/MP3D/test"

depth_map = np.load(os.path.join(data_path_est, "depth_up_est/" + file + "color.npy"))  # [:, :, 0]
depth_map = np.load(os.path.join(data_path_gt, "depth_up/" + file + "depth.npy"))[:, :, 0]

rgb_map = cv2.imread(os.path.join(data_path_gt, "image_up/" + file + "color.png"))

camera = Sphere(width=1024, height=512)
pcl_GT, color_GT = camera.depthmap2colorpcl(depth_map, rgb_map, format='rgb')

mask0 = pcl_GT[:, 0] < 2.5
mask1 = pcl_GT[:, 1] > -1
# mask2 = pcl_GT[:, 2] < 0
# mask3 = np.linalg.norm(pcl_GT, axis=1) > 0.2
mask4 = (abs(pcl_GT[:, 0]) > 0.2) | (abs(pcl_GT[:, 2]) > 0.2)

mask = mask1 * mask0 * mask4
pcl_GT = pcl_GT[mask, :]
color_GT = color_GT[mask, :]

viewer = setting_viewer(main_axis=False)
# camera_frame(view=viewer, pose=np.eye(4), size=0.5)

pcl_plotting_gt = setting_plc(view=viewer, size=0.2)
pcl_plotting_gt(pcl_GT, edge_color=color_GT / 255)

vispy.app.run()
print("end")
