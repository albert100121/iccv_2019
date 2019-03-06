import os
import numpy as np
from file_utilities import list_directories
import cv2
import matplotlib.pyplot as plt
from vispy_utilities import *
from CameraModels.Sphere import Sphere
import vispy

data_path = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/Stanford3D"
depth_map_dir = os.path.join(data_path, "depth_up")
rgb_map_dir = os.path.join(data_path, "image_up")
estimation_dir = os.path.join(data_path, "est")

depth_maps = list_directories(depth_map_dir)
rgb_maps = list_directories(rgb_map_dir)
estimation_maps = list_directories(estimation_dir)

i = 0
depth_map = np.load(os.path.join(depth_map_dir, depth_maps[i]))#[:, :, 0]
rgb_map = cv2.imread(os.path.join(rgb_map_dir, rgb_maps[i]))
estimation_map = np.load(os.path.join(estimation_dir, estimation_maps[i]))

plt.figure(1)
plt.imshow(depth_map, cmap='RdYlBu')
plt.colorbar()
plt.figure(2)
plt.imshow(estimation_map, cmap='RdYlBu')
plt.colorbar()
plt.figure(3)
# mask = depth_map > 0
# error = np.zeros_like(depth_map)
# error[mask] = (depth_map[mask] - estimation_map[mask]) * 100
error = (depth_map - estimation_map) * 100
mask = (error > 100.0) | (error < -100.0)
error[mask] = 0
plt.imshow(error, cmap="nipy_spectral")
plt.colorbar()

plt.show()
camera = Sphere(width=1024, height=512)
#
# pcl = camera.depthmap2pcl(depth_map)
pcl_GT, color_GT = camera.depthmap2colorpcl(depth_map, rgb_map, format='rgb')
pcl_EST, color_EST = camera.depthmap2colorpcl(estimation_map, rgb_map, format='rgb')
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
