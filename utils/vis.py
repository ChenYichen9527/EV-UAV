import os.path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import  cm
import cv2
import matplotlib
matplotlib.use('Agg')
#


def show_points_matplt(points,label):

    fig=plt.figure(figsize=(8, 6))
    ax = plt.axes(projection="3d")

    events_selected = points[label==1]
    events_no_selected = points[label==0]

    ax.scatter3D(events_selected[:, 0], events_selected[:, 1], events_selected[:, 2], '.', s=1, c='r', edgecolor='none')
    ax.scatter3D(events_no_selected[:, 0], events_no_selected[:, 1], events_no_selected[:, 2], '.', s=2, c='#A5A5A5', edgecolor='none')

    ax.set_xlim([0,350])
    ax.set_ylim([0, 260])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')
    plt.savefig('img.jpg', dpi=600, bbox_inches='tight', pad_inches=0)
    # plt.show()
    # plt.close()

if __name__ == '__main__':
    npz_root = r'/media/admin1/123/EV-UAV-dataset/test/test_021.npz'
    events = np.load(npz_root)
    evs_norm = events['evs_norm']
    evs_loc = events['ev_loc']
    points = evs_loc[:,0:3]
    label = evs_norm[:,4]
    show_points_matplt(points,label,0)



