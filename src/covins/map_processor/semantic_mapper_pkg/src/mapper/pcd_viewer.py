#!/usr/bin/python3

import numpy as np
import copy
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
# from rmf_cut import RMFCut
import struct
import time 
import yaml
import cv2


# Class to visualize open3D geometries 
class Viewer3D(object):

    def __init__(self, title: str):
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        self.main_vis = o3d.visualization.O3DVisualizer(title)
        app.add_window(self.main_vis)
        self.main_vis.add_action("Next class", self.class_switch_cb)

        self.class_idx = 5
        self.updated = True

        self.geometry_names = []
        self.geometries = {}
        self.trajectories = None

    # A function to draw class labels near points 
    def label_points(self):
        if self.labeled:
            return
        poses = self.point_cloud_o3d.points
        for i in range(0, len(poses), 80):
            pos = poses[i]
            col = self.rgb_load[i]
            closest_color = min(self.dataset.values(), key=lambda c: np.linalg.norm(col - c))
            class_name = next(label for label, c in self.dataset.items() if np.array_equal(c, closest_color))
            self.main_vis.add_3d_label(pos, class_name)
            self.labeled = True

    def setup_o3d_scene(self):
        # self.main_vis.add_geometry(self.point_cloud_o3d_name, self.point_cloud_o3d)
        self.main_vis.reset_camera_to_default()
        # center, eye, up
        self.main_vis.setup_camera(180,
                                [4, 2, 5],
                                [0, 0, -1.5],
                                [0, 1, 0])

    def update_o3d_scene(self):
        # print(self.geometry_names)
        for name in self.geometry_names:
            self.main_vis.remove_geometry(name)
            self.main_vis.add_geometry(name, self.geometries[name])

        # if self.show_esdf:
        #     self.main_vis.remove_geometry(self.raw_esdf_name)
        #     self.main_vis.add_geometry(self.raw_esdf_name, self.raw_esdf)
        # if self.show_ptc:
        #     self.main_vis.remove_geometry(self.raw_ptc_name)
        #     self.main_vis.add_geometry(self.raw_ptc_name, self.raw_ptc)

    def run_one_tick(self):
        app = o3d.visualization.gui.Application.instance
        tick_return = app.run_one_tick()
        if tick_return:
            self.main_vis.post_redraw()
        return tick_return

    # Gui button callback to increment the class index
    def class_switch_cb(self, instance=None):
        self.class_idx = (self.class_idx + 1)%150
        self.updated = True
        # reset geometries 
        # print(self.geometry_names)
        self.reset_geometry()
        
    def reset_geometry(self):
        for name in self.geometry_names:
            self.main_vis.remove_geometry(name)
        self.geometry_names = []
        self.geometries = {}
        
        if self.show_esdf:
            self.geometry_names.append(self.raw_esdf_name)
            self.geometries[self.raw_esdf_name] = self.raw_esdf
        if self.show_ptc:
            self.geometry_names.append(self.raw_ptc_name)
            self.geometries[self.raw_ptc_name] = self.raw_ptc

    def add_geometry(self, name:str, geometry):
        if name in self.geometry_names:
            print(f"WARN a geometry with this name ({name}) already exists. Updating instead. ")
            return
        self.geometries[name] = copy.deepcopy(geometry)
        self.geometry_names.append(name)
        # self.update_o3d_scene()

    def add_raw_esdf(self, name:str, geometry, show:bool):
        self.raw_esdf = geometry
        self.raw_esdf_name = name
        self.show_esdf = show
        if (show):
            self.main_vis.add_geometry(name, geometry)

    def add_raw_ptc(self, name:str, geometry, show:bool):
        self.raw_ptc = geometry
        self.raw_ptc_name = name
        self.show_ptc = show
        if (show):
            self.main_vis.add_geometry(name, geometry)

    def remove_geometry(self, name:str):
        # self.main_vis.remove_geometry(name)
        self.geometry_names.remove(name)
        del self.geometries[name]

    def update_geometry(self, name:str, geometry):
        if name not in self.geometry_names:
            print("WARN update_geometry is called with unregistered geometry name ")
            return
        self.geometries[name] = copy.deepcopy(geometry)
        self.update_o3d_scene()




       