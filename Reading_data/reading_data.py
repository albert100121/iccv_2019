import os
import numpy as np
from file_utilities import list_directories
import cv2
import matplotlib.pyplot as plt
from vispy_utilities import *
from CameraModels.Sphere import Sphere
import vispy

data_path = "/home/kike/Documents/Dataset/ICCV_stereo_data"
depth_map_dir = os.path.join(data_path, "depth_up")
rgb_map_dir = os.path.join(data_path, "image_up")
estimation_dir = os.path.join(data_path, "est")

depth_maps = list_directories(depth_map_dir)
rgb_maps = list_directories(rgb_map_dir)
estimation_maps = list_directories(estimation_dir)

i=1
depth_map = np.load(os.path.join(depth_map_dir, depth_maps[i]))[:, :, 0]
rgb_map = cv2.imread(os.path.join(rgb_map_dir, rgb_maps[i]))
estimation_map = np.load(os.path.join(estimation_dir, estimation_maps[i]))

fig1, ax1 = plt.subplots(1)
# plt.subplot(1, 3, 1)
ax1.imshow(depth_map, cmap='plasma')
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.grid(False)

fig2, ax2 = plt.subplots(1)
# plt.subplot(1, 3, 1)
ax2.imshow(rgb_map)
ax2.set_yticklabels([])
ax2.set_xticklabels([])
ax2.grid(False)

fig3, ax3 = plt.subplots(1)
# plt.subplot(1, 3, 1)
ax3.imshow(estimation_map, cmap='plasma')
ax3.set_yticklabels([])
ax3.set_xticklabels([])
ax3.grid(False)

# plt.show()
camera = Sphere(width=1024, height=512)

# pcl = camera.depthmap2pcl(depth_map)
pcl_GT, color_GT = camera.depthmap2colorpcl(depth_map, rgb_map, format='rgb')
pcl_EST, color_EST = camera.depthmap2colorpcl(estimation_map, rgb_map, format='rgb')

viewer = setting_viewer(main_axis=False)
camera_frame(view=viewer, pose=np.eye(4), size=0.5)
# camera_sphere(view=viewer, pose=np.eye(4), size=0.1, alpha=1)

# pcl_plotting_gt = setting_plc(view=viewer, size=0.1)
# pcl_plotting_gt(pcl_GT, edge_color=color_GT / 255)

# pcl_plotting_est = setting_plc(view=viewer, size=0.01)
# pcl_plotting_est(pcl_EST, edge_color=np.ones_like(pcl_EST))
#
pcl_plotting_est = setting_plc(view=viewer, size=1)
pcl_plotting_est(pcl_EST, edge_color=color_EST/255)

vispy.app.run()
print("end")
