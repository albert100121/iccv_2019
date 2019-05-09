import os, sys

sys.path.append(os.path.join(os.environ['HOME'], "Documents/PycharmProjects/Utilities"))
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from vispy_utilities import *
from CameraModels.Sphere import Sphere
import vispy
import numpy as np
import cv2

# from utils import *


data_main_path = "/home/kike/Documents/Dataset/ICCV_dataset/RealData_test/for_kike"
file = "smarthome1"

depth_map = np.load(os.path.join(data_main_path, "depth_up/" + file + ".npy"))  # [:, :, 0]
rgb_map = cv2.imread(os.path.join(data_main_path, "image_up/" + file + ".jpg"))

camera = Sphere(width=1024, height=512)
pcl_GT, color_GT = camera.depthmap2colorpcl(depth_map, rgb_map, format='rgb')

viewer = setting_viewer(main_axis=False)
# camera_frame(view=viewer, pose=np.eye(4), size=0.5)

pcl_plotting_gt = setting_plc(view=viewer, size=0.2)
pcl_plotting_gt(pcl_GT, edge_color=color_GT / 255)

vispy.app.run()
print("end")
