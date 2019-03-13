import os
import numpy as np
from file_utilities import list_directories
import cv2
import matplotlib.pyplot as plt
from vispy_utilities import *
from CameraModels.Sphere import Sphere
import vispy

data_path = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/PSM_ori_MP3D/eval"

dir_depth_gt = os.path.join(data_path, "depth_up_gt")
dir_depth_est = os.path.join(data_path, "depth_up_est")
dir_disp_gt = os.path.join(data_path, "disp_up_gt")
dir_disp_est = os.path.join(data_path, "disp_up_est")
dir_rgb_map = os.path.join(data_path, "image_up")

list_depth_gt = list_directories(dir_depth_gt)
list_depth_est = list_directories(dir_depth_est)
list_disp_gt = list_directories(dir_disp_gt)
list_disp_est = list_directories(dir_disp_est)
list_rgb_maps = list_directories(dir_rgb_map)

i = 0
depth_gt = np.load(os.path.join(dir_depth_gt, list_depth_gt[i]))[:, :, 0]
depth_est = np.load(os.path.join(dir_depth_est, list_depth_est[i]))

disp_gt = np.load(os.path.join(dir_disp_gt, list_disp_gt[i]))
mask = disp_gt > 20
disp_gt[mask] = 0
disp_est = np.load(os.path.join(dir_disp_est, list_disp_est[i]))
disp_est[mask] = 0
rgb_map = cv2.imread(os.path.join(dir_rgb_map, list_rgb_maps[i]))

plt.figure(1)
plt.title("Depth GT")
plt.imshow(depth_gt, cmap='RdYlBu')
plt.colorbar()
plt.savefig(os.path.join(data_path, "Depth_GT.png"))
plt.figure(2)
plt.title("Depth EST")
plt.imshow(depth_est, cmap='RdYlBu')
plt.colorbar()
plt.savefig(os.path.join(data_path, "Depth_EST.png"))
plt.figure(3)
plt.title("Disp GT")
plt.imshow(disp_gt, cmap='RdYlBu')
plt.colorbar()
plt.savefig(os.path.join(data_path, "Disp_GT.png"))
plt.figure(4)
plt.title("Disp EST")
plt.imshow(disp_est, cmap='RdYlBu')
plt.colorbar()
plt.savefig(os.path.join(data_path, "Disp_EST.png"))
plt.figure(5)
error = (depth_gt - depth_est) * 100
mask = (error > 100.0) | (error < -100.0)
error[mask] = 0
plt.title("Error depth")
plt.imshow(error, cmap="nipy_spectral")
plt.colorbar()
plt.savefig(os.path.join(data_path, "ErrorDepth.png"))
plt.show()
camera = Sphere(width=1024, height=512)
#
# pcl = camera.depthmap2pcl(depth_map)
pcl_GT, color_GT = camera.depthmap2colorpcl(depth_gt, rgb_map, format='rgb')
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
pcl_plotting_est = setting_plc(view=viewer, size=1)
pcl_plotting_est(pcl_EST, edge_color=color_EST / 255)
#
vispy.app.run()
print("end")
