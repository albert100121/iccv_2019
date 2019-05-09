import os
import numpy as np
import cv2
from CameraModels.Sphere import Sphere
from utils import *
"Area_3_lounge_1_7.png"
data_path = "/home/kike/Documents/Dataset/ICCV_dataset/SF3D/train"
rgb_map_file = "Area_1_office_31_5.png"
depth_map_file = "Area_1_office_31_5.npy"
disp_map_file = "Area_1_office_31_5.npy"

output_dir = "/home/kike/Documents/Dataset/ICCV_dataset/evaluation/GT_visualizations"


i = 0
depth_map = np.load(os.path.join(data_path, "depth_up/" + depth_map_file))#[:, :, 0]
rgb_map = cv2.imread(os.path.join(data_path, "image_up/" + rgb_map_file))
disp_map = np.load(os.path.join(data_path, "disp_up/" + disp_map_file))

disp = 20
mask = disp_map > disp
disp_map[mask] = disp

depth = 10
mask = depth_map > depth
depth_map[mask] = depth

plot_image(1, depth_map, "Depth GT", True, cmap='jet', filename=os.path.join(output_dir, "Depth_GT_SF3D.png"))
plot_image(3, disp_map, "Disp GT", True, cmap='jet', filename=os.path.join(output_dir, "Disp_GT_SF3D.png"))

plt.show()

# camera = Sphere(width=1024, height=512)
# pcl_GT, color_GT = camera.depthmap2colorpcl(depth_map, rgb_map, format='rgb')
#
# mask1 = pcl_GT[:, 1] > -1
# mask2 = pcl_GT[:, 0] > -1
# mask3 = np.linalg.norm(pcl_GT, axis=1) > 0.2
# mask4 = np.linalg.norm(pcl_GT, axis=1) < 1.5
#
# mask = mask1 * mask2 * mask3 * mask4
#
# pcl_GT = pcl_GT[mask, :]
# color_GT = color_GT[mask, :]
#
# viewer = setting_viewer(main_axis=False)
# pose = np.eye(4)
# pose[1, 3] = -0.5
# # sphere(view=viewer, pose=pose, size=0.2, alpha=1)
# pose[1, 3] = -1
# # sphere(view=viewer, pose=pose, size=0.2, alpha=1)
#
#
# pos = np.zeros((2, 3))
# pos[0, :] = [0, 0, 0]
# pos[1, :] = [0.2, 0.2, -0.5]
#
# pcl_plotting_gt = setting_plc(view=viewer, size=2)
# pcl_plotting_gt(pcl_GT, edge_color=color_GT / 255)
#
# vispy.app.run()
# print("end")
