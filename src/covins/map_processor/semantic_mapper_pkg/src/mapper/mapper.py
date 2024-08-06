#!/usr/bin/python3

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
# from rmf_cut import RMFCut
import struct
import time 
import yaml
import cv2

from .pcd_viewer import Viewer3D
from .mapper_3d import Mapper3D
from .mapper_2d import Mapper2D


class Mapper:
    """
        Class to handle the map generation and visualization
    """

    def __init__(self, vis=True):
        self.vis = vis
        self.m2d = Mapper2D(self)
        self.m3d = Mapper3D(self, vis)

        self.point_cloud_o3d_name = None
        self.point_cloud_o3d = None
        self.raw_point_cloud = None
        self.raw_esdf = None
        self.rgb_load = None
        self.trajectories = None

        if vis:
            self.viewer = Viewer3D("3D Mapper")
            # self.viewer.setup_o3d_scene()

    # Load the pointcloud and reverse color coding 
    def setup_point_clouds(self, ptc:o3d.geometry, name:str, z_deg:float, show:bool=True):
        # Check if ptc has points
        if len(ptc.points) == 0:
            print("Empty point cloud")
            return
        
        rgb_load = 255 - np.asarray(ptc.colors)*255
        rgb_load[rgb_load == 256] = 0       # Fix the issue with the 256 value
        corrected_colors = (rgb_load/255).astype(float)
        ptc.colors = o3d.utility.Vector3dVector(corrected_colors)
        rgb_load = rgb_load.astype(int)

        # Remove outliers from the point cloud
        # cl, ind = ptc.remove_radius_outlier(nb_points=8, radius=2)
        cl, ind = ptc.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.00001)
        print("Removed ", len(ptc.points) - len(cl.points), " outliers with radius outlier removal")
        ptc = ptc.select_by_index(ind)


        # Flip normals
        # ptc.normals = o3d.utility.Vector3dVector(-np.asarray(ptc.normals))
        # ptc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30), fast_normal_computation=True)
        # o3d.visualization.draw_geometries([ptc],  point_show_normal=True)

        #  around z axis
        z_angle = np.pi*(z_deg)/180   # -16 72 7 15
        rotation_matrix = np.array([[np.cos(z_angle), -np.sin(z_angle), 0],
                                    [np.sin(z_angle), np.cos(z_angle), 0],
                                    [0, 0, 1]])
        self.core_rotation_matrix = rotation_matrix
        ptc = ptc.rotate(rotation_matrix)

        # setup a point cloud 
        # the name is necessary to remove from the scene
        self.point_cloud_o3d_name = name
        self.point_cloud_o3d = ptc
        self.raw_point_cloud = ptc
        self.rgb_load = rgb_load[ind]

        self.build_point_classes()

        self.point_cloud_o3d = ptc
        self.raw_point_cloud = ptc

        if self.raw_esdf is not None:
            self.find_shared_bounds()

        if self.vis:
            self.viewer.add_raw_ptc(self.point_cloud_o3d_name, self.point_cloud_o3d, show)

    # Setup an ESDF map loaded as a o3d tensor
    def setup_intensity_map(self, ptc, name:str, z_deg:float, show:bool=True):
        # the name is necessary to remove from the scene
        self.esdf_name = name
        z_angle = np.pi*(z_deg)/180   # -16 72 7 15
        rotation_matrix = np.array([[np.cos(z_angle), -np.sin(z_angle), 0],
                                    [np.sin(z_angle), np.cos(z_angle), 0],
                                    [0, 0, 1]])
        if type(ptc) == o3d.core.Tensor:
            self.raw_esdf = ptc.rotate(o3d.core.Tensor(rotation_matrix), o3d.core.Tensor([0,0,0]) )
        elif type(ptc) == o3d.geometry.PointCloud:
            self.raw_esdf = ptc.rotate(rotation_matrix, [0,0,0] )

        self.esdf_rotation_matrix = rotation_matrix
        if self.raw_point_cloud is not None:
            self.find_shared_bounds()
        if self.vis:
            self.viewer.add_raw_esdf(self.esdf_name, ptc, show)
            
    # Find the shared bounds between the point cloud and the ESDF
    # DOESN"T ACCOUNT FOR ROTATION
    def find_shared_bounds(self):
        ptc_min_bounds = self.raw_point_cloud.get_min_bound() 
        ptc_max_bounds = self.raw_point_cloud.get_max_bound()
        esdf_min_bounds = self.raw_esdf.get_min_bound() #.numpy()
        esdf_max_bounds = self.raw_esdf.get_max_bound() #.numpy() tensor thing 
        
        min_bounds = np.minimum(ptc_min_bounds, esdf_min_bounds)
        max_bounds = np.maximum(ptc_max_bounds, esdf_max_bounds)
        
        
        
        self.shared_min_bound = min_bounds
        self.shared_max_bound = max_bounds
        
    
    # Parse the object labels and colors from a .yaml file
    def load_dataset(self, yaml_file:str):
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
        
        colors = data['colors']
        labels = data['labels']
        
        color_label_map = {}
        for idx, color in enumerate(colors):
            color_label_map[labels[idx]] = np.array(color)
        self.dataset = color_label_map
        
        # Append the void class 
        self.class_colors = np.concatenate(([[0,0,0]],colors))
        self.class_labels = np.concatenate((["empty"],labels))

        self.floor_classes = data["floor_ceil_ids"]
        self.obstacle_classes = data["obstacle_ids"]
        self.dynamic_classes = data["dynamic_ids"]

        # Init class id for 'filter_ptc' 
        self.class_idx = 2

        self.updated = True
        self.labeled = False

    # Build the class id vector based on self.raw_point_cloud. Need to load_dataset first
    def build_point_classes(self):
        rgb_load = (np.asarray(self.raw_point_cloud.colors)*255).astype(int)
        point_classes = np.zeros((rgb_load.shape[0], 1))
        for idx, color in enumerate(rgb_load):
            distances = np.sqrt(np.sum((self.class_colors - color)**2, axis=1))
            # Find the index of the closest color
            class_index = np.argmin(distances)
            point_classes[idx] = class_index
        self.point_classes = point_classes
        
        print("Constructed a point class array of shape: ", point_classes.shape)
        print("Found classes: ", np.unique(point_classes))


    # Setup a trajectory point cloud
    def load_trajectories(self, trajectories:list, z_deg:float):
        z_angle = np.pi*(z_deg)/180
        rotation_matrix = np.array([[np.cos(z_angle), -np.sin(z_angle), 0],
                                [np.sin(z_angle), np.cos(z_angle), 0],
                                [0, 0, 1]])
        scale = 1.
        shift = np.array([-4.9, -2, 0])

        all_trajectories = o3d.geometry.PointCloud()
        for trajectory in trajectories:
            traj = o3d.io.read_point_cloud(trajectory)
            all_trajectories += traj

        all_trajectories = all_trajectories.scale(scale, center=all_trajectories.get_center())
        all_trajectories = all_trajectories.rotate(rotation_matrix) #self.core_rotation_matrix
        all_trajectories = all_trajectories.translate(shift)

        self.trajectories = all_trajectories
        
        
    def val_esdf(self):
        esdf_grid, _, _ = self.m2d.construct_2d_esdf( 'mean')
        # mpr.m2d.fuse_pointclouds(tf)
        flat_esdf = self.m3d.test_esdf_voxel_grid(0.2, True)  
                  
        difference = np.zeros_like(esdf_grid)
        difference[esdf_grid != -2] = flat_esdf[esdf_grid != -2] - esdf_grid[esdf_grid != -2]
        avg_diff = np.mean(difference[difference != 0])
        print("Average difference: ", avg_diff)
        # Compare
        plt.subplot(1, 3, 1)
        plt.imshow(esdf_grid, vmin=-2, vmax=5)
        plt.subplot(1, 3, 2)
        plt.imshow(flat_esdf, vmin=-2, vmax=5)
        plt.subplot(1, 3, 3)
        plt.imshow(difference)
        plt.show()

    def compare_esdf_compression(self):
        # Compare esdf build methods
        esdf_grid_maa, _, _  = self.m2d.construct_2d_esdf( 'max_abs' )
        esdf_grid_mean, _, _ = self.m2d.construct_2d_esdf( 'mean' )
        esdf_grid_max, _, _  = self.m2d.construct_2d_esdf( 'max' )

        # esdf_grid_max = self.m3d.test_esdf_voxel_grid(0.2, filters=True)  

        cmap = plt.cm.viridis  # Choose any colormap that you want
        cmap.set_under('white') 

        fig = plt.figure()

        # Display images
        ax1 = plt.subplot(231)
        ax1.tick_params(which = 'both', size = 0, labelsize = 0)
        plt.imshow(esdf_grid_maa, vmax=2, vmin=-1.9, cmap=cmap)
        plt.title('max_abs', fontsize=14)

        ax2 = plt.subplot(232)
        ax2.tick_params(which = 'both', size = 0, labelsize = 0)
        plt.imshow(esdf_grid_mean, vmax=2, vmin=-2, cmap=cmap)
        plt.title('mean', fontsize=14)

        ax3 = plt.subplot(233)
        ax3.tick_params(which = 'both', size = 0, labelsize = 0)
        plt.imshow(esdf_grid_max, vmax=2, vmin=-2, cmap=cmap)
        # plt.colorbar()
        plt.title('max', fontsize=14)

        global_min = -2
        global_max = 2

        # Plot histograms
        ax4 = plt.subplot(234)
        n, bins, patches = plt.hist(esdf_grid_maa[esdf_grid_maa!=-2].flatten(), bins=50)
        for c, p in zip(bins, patches):
            norm_color = (c - global_min) / (global_max - global_min)
            plt.setp(p, 'facecolor', cmap(norm_color))

        plt.subplot(235)
        n, bins, patches = plt.hist(esdf_grid_mean[esdf_grid_maa!=-2].flatten(), bins=50)
        for c, p in zip(bins, patches):
            norm_color = (c - global_min) / (global_max - global_min)
            plt.setp(p, 'facecolor', cmap(norm_color))

        plt.subplot(236)
        n, bins, patches = plt.hist(esdf_grid_max[esdf_grid_maa!=-2].flatten(), bins=50)
        for c, p in zip(bins, patches):
            norm_color = (c - global_min) / (global_max - global_min)
            plt.setp(p, 'facecolor', cmap(norm_color))

        # Shared x-label
        fig.text(0.5, 0.04, 'Pixel intensity', ha='center', fontsize=14)

        # plt.tight_layout()
        plt.show()
        
    def test_cluster_params(self):
        # Range w_spat and w_int from 0 to 1 in 0.1 increments
        images = []
        spat_range = np.arange(0.1, 0.6, 0.1)
        int_range = np.arange(0.1, 2.1, 0.2)
        esdf_grid, count_grid, height_grid = self.m2d.construct_2d_esdf(compression="max_abs")
        for w_spat in spat_range:
            for w_int in int_range:
                # Run the clustering algorithm
                image, _, _ = self.m2d.cluster_dbscan(False, w_spat, w_int, esdf_grid, count_grid, height_grid)
                
                images.append(image)
                print(f"Finished w_spat: {w_spat:.2f}, w_int: {w_int:.2f}")  
                              
        # Display the images
        fig, axs = plt.subplots(len(spat_range), len(int_range))
        for i in range(len(spat_range)):
            for j in range(len(int_range)):
                axs[i,j].imshow(images[i*len(int_range) + j])
                axs[i,j].axis('off')
                axs[i,j].set_title(f"w_s: {spat_range[i]:.2f}, w_i: {int_range[j]:.2f}")
            
        plt.tight_layout()
        plt.show()


    def get_unique_clusters(self, image):
        """ Return a list of unique values representing different clusters. """
        return np.unique(image.reshape(-1, image.shape[2]), axis=0)

    def calculate_iou(self, image1, image2):
        """ Calculate Intersection over Union (IoU) score for two binary images. """
        intersection = np.logical_and(image1, image2).sum()
        union = np.logical_or(image1, image2).sum()
        if union == 0:
            return 0
        return intersection / union

    def find_max_iou(self, gt_map, max_shift=20):
        """ Calculate IoU for different shifts of one image over another. """
        base_image, _, _ = self.m2d.cluster_dbscan(False, ) # 0.5, 0.8
        base_image = base_image > 0

        # Bring both images to the same shape conserving the scale
        max_shape = np.maximum(gt_map.shape, base_image.shape)
        
        rows, cols = base_image.shape
        iou_scores = np.zeros((2 * max_shift + 1, 2 * max_shift + 1))
        for i in range(-max_shift, max_shift + 1):
            for j in range(-max_shift, max_shift + 1):
                shifted_image = np.roll(gt_map, shift=(i, j), axis=(0, 1))
                iou_score = self.calculate_iou(base_image, shifted_image)
                iou_scores[i + max_shift, j + max_shift] = iou_score

        # Find the maximum IoU and its corresponding shift
        max_iou = np.max(iou_scores)
        max_index = np.unravel_index(np.argmax(iou_scores), iou_scores.shape)
        max_shifts = (max_index[0] - max_shift, max_index[1] - max_shift)

        return iou_scores, max_shifts, max_iou
            
    def calculate_iou_per_cluster(self, ground_truth, segmentation, ground_truth_labels, segmentation_labels):
        """ Calculate IoU for each cluster pair. """
        # Create dictionaries to hold pixel positions for each label
        gt_clusters = {tuple(label): np.where((ground_truth == label).all(axis=2)) for label in ground_truth_labels}
        seg_clusters = {label: np.where(segmentation == label) for label in segmentation_labels}
        
        # Dictionary to store IoU scores
        iou_scores = {}
        
        # Calculate IoU for each pair of clusters
        for gt_label, gt_pixels in gt_clusters.items():
            for seg_label, seg_pixels in seg_clusters.items():
                # Find intersection and union of the two clusters
                gt_set = set(zip(*gt_pixels))
                seg_set = set(zip(*seg_pixels))
                intersection = gt_set.intersection(seg_set)
                union = gt_set.union(seg_set)
                
                if union:
                    iou_score = len(intersection) / len(union)
                    iou_scores[(gt_label, seg_label)] = iou_score
        
        return iou_scores
    

    def draw_nn_perf(self):
        import matplotlib.lines as mlines
        from matplotlib import colormaps
        # A figure to show how different neural networks compare
        # ADE20k mIOU / FLOPs

        ## Transformers 
        # EfficientViT:
        evit_gmacs = [3.1, 9.1, 22, 36, 45]
        evit_miou = [43, 46, 49, 49.2, 50.7]
        evit_mparams = [4.8, 15, 39, 40, 51]

        # SegFormer:   (from EffViT)
        segformer_gmacs = [ 16, 62, 96 ]
        segformer_miou = [ 42.2, 46.5, 50.3 ] 
        segformer_mparams = [ 14, 46.5, 50.3 ]
        
        # Mask2Former (og): Swin-S, Swin-B, Swin-L  # R50, R101,
        m2f_gmacs = [ 232//2,313//2, 466//2, 868//2 ] # 226//2, 232//2,
        m2f_miou = [  47.4, 52.3, 52.4, 56.4]     # 47.2, 47.7,
        m2f_mparams = [ 47, 69, 107, 217]  #44, 63,
        
        # SWIN (T, S, B)
        swin_gmacs = [ 945//2, 1038//2, 1188//2 ]
        swin_miou = [ 44.5, 47.6, 48.1 ]
        swin_mparams = [ 60, 80, 121 ]
        
        # BEIT-3: pwc
        beit3_gmacs = [ ]
        beit3_miou = [62.8]
        beit3_mparams = [1310]
        
        # ViT-Adapter-L: pwc
        # vitl_gmacs = [ ]
        # vitl_miou = [61.5 ]
        # vitl_mparams = [571]
        
        ## CNNs
        # InternImage (T,S,B)
        internim_gmacs = [944//2, 1017//2, 1185//2]
        internim_miou = [47.9, 50.1, 50.8]
        internim_mparams = [ 30,  50,  97]  
         
        # SegNetXT:  (from EffViT)
        segnetxt_gmacs = [ 6.6, 16, 35]
        segnetxt_miou = [ 41.1, 44.3, 48.5]
        segnetxt_mparams = [ 4.3, 14, 28]

        # FCN:
        fcn_gmacs = [ 276//2] 
        fcn_miou = [ 41.4 ] 

        # DeepLabV3+:
        DeeplabV3_gmacs = [176//2, 255//2] #[ ]
        DeeplabV3_miou = [41.5, 43.2] #[  ]
        DeeplabV3_mparams = [ 43.6, 62.7 ]  #[62.7]
        
        # PSPNet:
        pspnet_gmacs = [ 256//2 ]
        pspnet_miou = [ 44.4 ]

        ## Hybrids
        # MobileVIT: 
        mvit_gmacs = [0.75, 1, 1.8]
        mvit_miou = [33.5, 36.4, 39.1]
        mvit_mparams = [6.3, 9.7, 13.6]

        plot_type = "gmacs"
        plt.style.use('seaborn-poster')
        # Define colors for each group
        transformers_color = colormaps["winter"](np.linspace(0.2, 0.8, 5))
        cnns_color = colormaps["autumn"](np.linspace(0.2, 0.8, 5))
        fig, ax = plt.subplots()
        if plot_type == "gmacs":
            line1, = ax.plot(evit_gmacs, evit_miou, label="Transformers:\n  EfficientViT", marker='o', linewidth=4, markersize=10, linestyle='--', color=transformers_color[4])
            line2, = ax.plot(segformer_gmacs, segformer_miou, label="SegFormer", marker='o', linewidth=2, markersize=10, linestyle='--', color=transformers_color[3])
            line3, = ax.plot(m2f_gmacs, m2f_miou, label="Mask2Former", marker='o',   linewidth=2, markersize=10, linestyle='--', color=transformers_color[2]) 
            line4, = ax.plot(mvit_gmacs, mvit_miou, label="Hybrid:\n  MobileVIT", marker='o', linewidth=2, markersize=10, linestyle='--', color=transformers_color[1])
            # line5, = ax.plot(swin_gmacs, swin_miou, label="SWIN", marker='o', linewidth=2, markersize=10, color=transformers_color[0])
            line9, = ax.plot(segnetxt_gmacs, segnetxt_miou, label="CNNs:\n  SegNetXT", marker='D', linewidth=2, markersize=10, color=cnns_color[4])
            line6, = ax.plot(fcn_gmacs, fcn_miou, label="FCN", marker='D', linewidth=2, markersize=10, color=cnns_color[3])
            line7, = ax.plot(DeeplabV3_gmacs, DeeplabV3_miou, label="DeepLabV3+", marker='D', linewidth=2, markersize=10, color=cnns_color[2])
            line8, = ax.plot(pspnet_gmacs, pspnet_miou, label="PSPNet", marker='D', linewidth=2, markersize=10, color=cnns_color[1])
            # line10, = ax.plot(internim_gmacs, internim_miou, label="InternImage", marker='D', linewidth=2, markersize=10, color=cnns_color[0])
            ax.set_xlabel("Compute (GMACs)",) # 
            # Create legendsl ine5 line10
            legend1 = plt.legend([line1, line2, line3, line4, ], ["EfficientViT", "SegFormer", "Mask2Former", "MobileViT", "SWIN"], fontsize=14, loc=(0.65, 0.32))
            legend2 = plt.legend([line9, line6, line7, line8, ], ["SegNetXT", "FCN", "DeepLabV3+", "PSPNet", "InternImage"], fontsize=14, loc=(0.65, 0.02))

        elif plot_type == "params":
            line1, = ax.plot(evit_mparams, evit_miou, label="Transformers:\n  EfficientViT", marker='o', linewidth=4, markersize=10, linestyle='-', color=transformers_color[4])
            line2, = ax.plot(segformer_mparams, segformer_miou, label="SegFormer", marker='o', linewidth=2, markersize=10, linestyle='-', color=transformers_color[3])
            line3, = ax.plot(m2f_mparams, m2f_miou, label="Mask2Former", marker='o',   linewidth=2, markersize=10, linestyle='-', color=transformers_color[2]) 
            line4, = ax.plot(mvit_mparams, mvit_miou, label="Hybrid:\n  MobileVIT", marker='o', linewidth=2, markersize=10, color=transformers_color[1])
            line5, = ax.plot(swin_mparams, swin_miou, label="SWIN", marker='o', linewidth=2, markersize=10, color=transformers_color[0])
            line9, = ax.plot(segnetxt_mparams, segnetxt_miou, label="CNNs:\n  SegNetXT", marker='D', linewidth=2, markersize=10, color=cnns_color[4])
            # line6, = ax.plot(fcn_mparams, fcn_miou, label="FCN", marker='D', linewidth=2, markersize=10, color=cnns_color[3])
            line7, = ax.plot(DeeplabV3_mparams, DeeplabV3_miou, label="DeepLabV3+", marker='D', linewidth=2, markersize=10, color=cnns_color[2])
            # line8, = ax.plot(pspnet_mparams, pspnet_miou, label="PSPNet", marker='D', linewidth=2, markersize=10, color=cnns_color[1])
            line10, = ax.plot(internim_mparams, internim_miou, label="InternImage", marker='D', linewidth=2, markersize=10, color=cnns_color[0])
            ax.set_xlabel("Model Parameters (M)",) #          
            # Create legends
            legend1 = plt.legend([line1, line2, line3, line4, line5], ["EfficientViT", "SegFormer", "Mask2Former", "MobileViT", "SWIN"], fontsize=14, loc=(0.8, 0.22))
            legend2 = plt.legend([line9, line7, line10], ["SegNetXT",  "DeepLabV3+",  "InternImage"], fontsize=14, loc=(0.8, 0.02))
           
        ax.grid()
        # Set x to log scale
        # ax.set_xscale('log')
        ax.set_ylabel("Accuracy (mIoU% on ADE20k val)")
        ax.tick_params(axis='both', which='major', )  #labelsize=12 Increase the tick siz
              
        # Set legend titles with increased font size and bold weight
        legend1.set_title("Transformers", prop={"size": 16, "weight": "bold"})
        legend2.set_title("CNNs", prop={"size": 16, "weight": "bold"})
        # Add legends to the plot
        ax.add_artist(legend1)
        ax.add_artist(legend2)
        
        plt.show()
