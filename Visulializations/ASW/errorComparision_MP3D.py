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

data_path_gt = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/Visualization/reasonable_image_MP3D"
data_path_est = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/Visualization/evaluation/ASW_MP3D/eval"

dir_depth_gt = os.path.join(data_path_gt, "depth_up_gt")
dir_disp_gt = os.path.join(data_path_gt, "disp_up_gt")
dir_disp_est = os.path.join(data_path_est, "disp_up_est")
dir_rgb_map = os.path.join(data_path_gt, "image_up")

list_depth_gt = list_directories(dir_depth_gt)
list_disp_gt = list_directories(dir_disp_gt)
list_disp_est = list_directories(dir_disp_est)
list_rgb_maps = list_directories(dir_rgb_map)

i = 0
depth_gt = np.load(os.path.join(dir_depth_gt, list_depth_gt[i]))[:, :, 0]
disp_gt = np.load(os.path.join(dir_disp_gt, list_disp_gt[i]))

im = Image.open(os.path.join(dir_disp_est, list_disp_est[i]))
imarray = np.array(im) * (-1)
disp_est = imutils.rotate_bound(imarray, 90) * 180 / 512
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


rgb_map = cv2.imread(os.path.join(dir_rgb_map, list_rgb_maps[i]))

plot_image(1, depth_gt, "Depth GT", True, cmap='gnuplot2', filename=os.path.join(data_path_est, "Depth_GT.png"))
plot_image(2, depth_est, "Depth EST", True, cmap='gnuplot2', filename=os.path.join(data_path_est, "Depth_EST.png"))

plot_image(3, disp_gt, "Disp GT", True, cmap='gnuplot2', filename=os.path.join(data_path_est, "Disp_GT.png"))
plot_image(4, disp_est, "Disp EST", True, cmap='gnuplot2',filename=os.path.join(data_path_est, "Disp_EST.png"))

error = abs(depth_gt - depth_est)
mask = (error > 2)
error[mask] = 2

plot_image(5, error, "Error depth", True, cmap="hot", filename=os.path.join(data_path_est, "ErrorDepth.png"))

plt.show()
# camera = Sphere(width=1024, height=512)
# #
# # pcl = camera.depthmap2pcl(depth_map)
# pcl_GT, color_GT = camera.depthmap2colorpcl(depth_gt, rgb_map, format='rgb')
# pcl_EST, color_EST = camera.depthmap2colorpcl(depth_est, rgb_map, format='rgb')
# #
# viewer = setting_viewer(main_axis=False)
# # # camera_frame(view=viewer, pose=np.eye(4), size=0.5)
# # # camera_sphere(view=viewer, pose=np.eye(4), size=0.1, alpha=1)
# #
# # pcl_plotting_gt = setting_plc(view=viewer, size=0.1)
# # pcl_plotting_gt(pcl_GT, edge_color=color_GT / 255)
#
# # pcl_plotting_est = setting_plc(view=viewer, size=0.01)
# # pcl_plotting_est(pcl_EST, edge_color=np.ones_like(pcl_EST))
# # #
# pcl_plotting_est = setting_plc(view=viewer, size=1)
# pcl_plotting_est(pcl_EST, edge_color=color_EST / 255)
# #
# vispy.app.run()
# print("end")