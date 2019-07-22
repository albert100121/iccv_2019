import os
import numpy as np
from file_utilities import list_directories
import cv2
from vispy_utilities import *
from CameraModels.Sphere import Sphere
import vispy
from utils import *
from PIL import Image
import imutils
from CV_stereo_matching.opencv_stereo_matching import StereoMatching


report_name = "MP3D_StereoMatching_rms_mae.csv"
data_path_gt = "/home/kike/Documents/Dataset/ICCV_dataset/MP3D/test"

# report_name = "SF3D_StereoMatching_rms_mae.csv"
# data_path_gt = "/home/kike/Documents/Dataset/ICCV_dataset/SF3D/test"

dir_disp_gt = os.path.join(data_path_gt, "disp_up")
dir_depth_gt = os.path.join(data_path_gt, "depth_up")
dir_rgb_map_up = os.path.join(data_path_gt, "image_up")
dir_rgb_map_down = os.path.join(data_path_gt, "image_down")

list_disp_gt = list_directories(dir_disp_gt)
list_depth_gt = list_directories(dir_depth_gt)
list_rgb_map_up = list_directories(dir_rgb_map_up)
list_rgb_map_down = list_directories(dir_rgb_map_down)

i = 0
stereo = StereoMatching()

depth_gt = np.load(os.path.join(dir_depth_gt, list_depth_gt[i]))[:, :, 0]
disp_gt = np.load(os.path.join(dir_disp_gt, list_disp_gt[i]))

rgb_map_up = cv2.imread(os.path.join(dir_rgb_map_up, list_rgb_map_up[i]))
rgb_map_down = cv2.imread(os.path.join(dir_rgb_map_down, list_rgb_map_down[i]))

imgL = imutils.rotate_bound(rgb_map_up, 270)
imgR = imutils.rotate_bound(rgb_map_down, 270)
disp_est = stereo.run(imgL, imgR) * 180 / 512
disp_est = imutils.rotate_bound(disp_est, 90)
depth_est = disp2depth(0.2, disp_est)

disp = 20
mask = disp_gt > disp
disp_gt[mask] = disp
mask = disp_est > disp
disp_est[mask] = disp

depth = 10
mask = depth_est > depth
depth_est[mask] = depth
mask = depth_gt > depth
depth_gt[mask] = depth

rgb_map = cv2.imread(os.path.join(dir_rgb_map_up, list_rgb_map_up[i]))

error = abs(depth_gt - depth_est)
mask = (error > 2)
error[mask] = 2

camera = Sphere(width=1024, height=512)
#
# pcl = camera.depthmap2pcl(depth_map)
# pcl_GT, color_GT = camera.depthmap2colorpcl(depth_gt, rgb_map, format='rgb')
pcl_EST, color_EST = camera.depthmap2colorpcl(depth_est, rgb_map, format='rgb')
#
viewer = setting_viewer(main_axis=False)
# # camera_frame(view=viewer, pose=np.eye(4), size=0.5)
# # camera_sphere(view=viewer, pose=np.eye(4), size=0.1, alpha=1)
#
# pcl_plotting_gt = setting_plc(view=viewer, size=0.1)
# pcl_plotting_gt(pcl_GT, edge_color=color_GT / 255)

# pcl_plotting_est = setting_plc(view=viewer, size=0.01)
# pcl_plotting_est(pcl_EST, edge_color=np.ones_like(pcl_EST))
# #
pcl_plotting_est = setting_plc(view=viewer, size=0.01)
pcl_plotting_est(pcl_EST, edge_color=color_EST / 255)
#
vispy.app.run()
print("end")
