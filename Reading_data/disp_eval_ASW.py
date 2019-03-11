import os
import numpy as np
from file_utilities import list_directories
import cv2
from scipy.misc import imsave
from vispy_utilities import *
from CameraModels.Sphere import Sphere
import imutils
import matplotlib.pyplot as plt
from PIL import Image
import math
import vispy


def disp2depth(baseline, disp):
    angle = np.zeros((512, 1024))
    angle2 = np.zeros((512, 1024))
    for i in range(1024):
        for j in range(512):
            theta_T = math.pi - ((j + 0.5) * math.pi / 512)
            angle[j, i] = baseline * math.sin(theta_T)
            angle2[j, i] = baseline * math.cos(theta_T)

    mask = disp > 0

    depth = np.zeros_like(disp).astype(np.float)
    depth[mask] = (angle[mask] / np.tan(disp[mask] / 180 * math.pi)) + angle2[mask]
    return depth


data_path = "/home/kike/Documents/Dataset/ICCV_dataset/MP3D/test"
rgb_map_dir = os.path.join(data_path, "image_up")
disp_map_gt_dir = os.path.join(data_path, "disp_up")
disp_map_est_dir = os.path.join(data_path, "asw_estimation")


rgb_maps = list_directories(rgb_map_dir)
disp_maps_gt = list_directories(disp_map_gt_dir)
disp_maps_est = list_directories(disp_map_est_dir, key="_pp")

for i in range(len(disp_maps_est)):
    # i = 1
    rgb_map = cv2.imread(os.path.join(rgb_map_dir, rgb_maps[i]))
    disp_gt = np.load(os.path.join(disp_map_gt_dir, disp_maps_gt[i]))

    mask = disp_gt > 192
    disp_gt[mask] = 0

    im = Image.open(os.path.join(disp_map_est_dir, disp_maps_est[i]))
    imarray = np.array(im) * (-1)
    disp_asw = imutils.rotate_bound(imarray, 90) * 180 / 512

    # plt.figure(1)
    # plt.subplot(1, 3, 1)
    # plt.title("disparity GT")
    # plt.imshow(disp_gt, cmap='RdYlBu')
    # plt.colorbar()
    #
    # # plt.figure(2)
    # plt.subplot(1, 3, 2)
    # plt.title("disparity ASW")
    # plt.imshow(disp_asw, cmap='RdYlBu')
    # plt.colorbar()
    #
    # error = disp_gt - disp_asw
    # mask = abs(error) > 40
    # error[mask] = 0
    #
    # # plt.figure(3)
    # plt.subplot(1, 3, 3)
    # plt.title("disparity Error")
    # plt.imshow(error, cmap="nipy_spectral")
    # plt.colorbar()

    depth_gt = disp2depth(0.2, disp_gt)
    depth_asw = disp2depth(0.2, disp_asw)
    mask = depth_asw > depth_gt.max()
    depth_asw[mask] = 0

    plt.figure(2)
    plt.subplot(1, 3, 1)
    plt.title("depth GT")
    plt.imshow(depth_gt, cmap='RdYlBu')
    plt.colorbar()

    # plt.figure(2)
    plt.subplot(1, 3, 2)
    plt.title("depth ASW")
    plt.imshow(depth_asw, cmap='RdYlBu')
    plt.colorbar()

    error = depth_gt - depth_asw
    # plt.figure(3)
    plt.subplot(1, 3, 3)
    plt.title("depth Error")
    plt.imshow(error, cmap="nipy_spectral")
    plt.colorbar()

    plt.show()

#
# camera = Sphere(width=1024, height=512)
#
# pcl_GT, color_GT = camera.depthmap2colorpcl(depth_gt, rgb_map, format='rgb')
# pcl_EST, color_EST = camera.depthmap2colorpcl(depth_asw, rgb_map, format='rgb')
# #
# viewer = setting_viewer(main_axis=False)
# # # camera_frame(view=viewer, pose=np.eye(4), size=0.5)
# # # camera_sphere(view=viewer, pose=np.eye(4), size=0.1, alpha=1)
# #
# # pcl_plotting_gt = setting_plc(view=viewer, size=0.1)
# # pcl_plotting_gt(pcl_GT, edge_color=color_GT / 255)
#
# pcl_plotting_est = setting_plc(view=viewer, size=1)
# pcl_plotting_est(pcl_EST, edge_color=color_EST / 255)
#
#
# # pcl_plotting_est = setting_plc(view=viewer, size=0.01)
# # pcl_plotting_est(pcl_EST, edge_color=np.ones_like(pcl_EST))
# # #
# #
# vispy.app.run()
# print("end")
