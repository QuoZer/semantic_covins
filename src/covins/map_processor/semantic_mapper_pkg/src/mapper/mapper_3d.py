#!/usr/bin/python3

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.ndimage import distance_transform_edt

# from rmf_cut import RMFCut
import struct
import time 
import yaml
import cv2

from .pcd_viewer import Viewer3D
# from mapper import Mapper


class Mapper3D:
    """
        Class to handle the 3D map generation and visualization
    """

    def __init__(self, mapper, vis:bool = True):
        self.vis = vis
        self.mapper = mapper

        self.updated = True
        # if vis:
        #     self.mapper.viewer = mapper.viewer   
    
    # Assemble a pointcloud of classes lying on the walls based on self.raw_point_cloud
    def merge_wall_classes(self, wall_classes=[1, 9, 15, 23], obj_outlier_removal=False) -> o3d.geometry.PointCloud:
        # Walls, windows, doors, paintings,  

        if self.mapper.point_classes is None:
            print("No point classes found")
            raise ValueError("No point classes found")
        
        # Get the inliers
        class_indices = []
        for class_id in wall_classes:
            class_indices.append(np.where(self.mapper.point_classes == class_id)[0])
            
        # Merge the class indices and Remove duplicates
        merged_indices = np.unique(np.concatenate(class_indices))
        
        # Create a new point cloud with the merged classes
        merged_wall_classes = self.mapper.raw_point_cloud.select_by_index(merged_indices, invert=False)
        
        if not obj_outlier_removal:
            return merged_wall_classes
        
        # Iterate over all classes and remove wall points inside the objects
        np_points = np.asarray(merged_wall_classes.points)
        points_to_remove = []
        skipped_class_names = []
        for class_id in range(2, len(self.mapper.dataset)):
            # Skip the wall classes
            if class_id in wall_classes:
                continue

            class_indices, labels = self.cluster_class(class_id, eps=0.5, min_points=8)
            class_name = self.mapper.class_labels[class_id]

            class_entries = class_indices.shape[0] 
            if (class_entries <= 10):
                skipped_class_names.append(class_name)
                continue

            for cluster in np.unique(labels):
                cluster_indices = np.where(labels == cluster)[0]
                cluster_ptc  = self.mapper.raw_point_cloud.select_by_index(class_indices[cluster_indices], invert=False)
                aa_bb = cluster_ptc.get_axis_aligned_bounding_box()
                # aa_bb = self.part_aligned_bounding_box(cluster_ptc)
                # aa_bb.color = self.dataset[class_name]/255
                
                # Reduce the bb by 10% 
                extent = aa_bb.get_extent()*0/ 2
                box_min = aa_bb.get_min_bound() + extent
                box_max = aa_bb.get_max_bound() - extent
                
                
                # Find indicies of points inside the bounding box
                inside_indices = np.where((np_points[:, 0] >= box_min[0]) & (np_points[:, 0] <= box_max[0]) &
                                        (np_points[:, 1] >= box_min[1]) & (np_points[:, 1] <= box_max[1]) &
                                        (np_points[:, 2] >= box_min[2]) & (np_points[:, 2] <= box_max[2]))[0]
                points_to_remove.append(inside_indices)
        
        uniqie_points_to_remove = np.unique(np.concatenate(points_to_remove))
        print(f"Removing {len(uniqie_points_to_remove)} points")
        print("Skipped classes: ", skipped_class_names)

        part_point_classes = self.mapper.point_classes[merged_indices]
        removed_point_classes = part_point_classes[uniqie_points_to_remove]
        classes, counts = np.unique(removed_point_classes, return_counts=True)
        print("Removed classes: ", dict(zip(classes, counts)))


        merged_wall_classes = merged_wall_classes.select_by_index(uniqie_points_to_remove, invert=True)        
        
        return merged_wall_classes
    
    # Draw histograms for x and y coordinates with point classes
    def draw_class_distribution(self):
        np_points = np.asarray(self.mapper.raw_point_cloud.points)
        classes = self.mapper.point_classes
        bin_size = 0.3
        font_size = 20
        
        # Extract x and y coordinates from the point cloud
        x_coordinates = np_points[:, 0]
        y_coordinates = np_points[:, 1]

        # Determine the range of x and y coordinates
        x_min, x_max = np.min(x_coordinates), np.max(x_coordinates)
        y_min, y_max = np.min(y_coordinates), np.max(y_coordinates)

        # Determine the number of bins based on the range of coordinates
        num_bins_x = int((x_max - x_min)/bin_size) + 1
        num_bins_y = int((y_max - y_min)/bin_size) + 1

        # Initialize arrays to store class counts for each bin
        class_counts_x = np.zeros((num_bins_x, 150))
        class_counts_y = np.zeros((num_bins_y, 150))

        # Iterate through each point and increment the corresponding bin
        for i in range(len(np_points)):
            x_idx = int((np_points[i, 0] - x_min)/bin_size)
            y_idx = int((np_points[i, 1] - y_min)/bin_size)
            class_idx = int(classes[i])
            if class_idx-1 in self.mapper.obstacle_classes:
                class_idx = 1
            else:
                class_idx = 9
            class_counts_x[x_idx, class_idx] += 1
            class_counts_y[y_idx, class_idx] += 1

        # Plot histograms for x-axis and y-axis
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

        # Plot histogram for x-axis
        for class_idx in range(len(np.unique(classes))):
            ax = axes[0]
            bottom = np.sum(class_counts_x[:, :class_idx], axis=1)
            bar = ax.bar(range(num_bins_x), class_counts_x[:, class_idx], bottom=bottom, label=f'Class {class_idx}',
                color=f'C{class_idx}', alpha=0.7)
            if class_idx == 1:
                bar.set_label("Walls")
            elif class_idx == 9:
                bar.set_label("Objects")
            else:
                bar.set_label("__Other")
            ax.tick_params(axis='both', which='major', labelsize=font_size)
            # ax.set_xticks(range(num_bins_x))
            # ax.set_xticklabels([f'{i+x_min}' for i in range(num_bins_x)])
            ax.set_title('X-axis', weight='bold',fontsize=font_size)
            ax.legend(fontsize=font_size, loc='upper left')

        # Plot histogram for y-axis
        for class_idx in range(len(np.unique(classes))):
            ax = axes[1]
            bottom = np.sum(class_counts_y[:, :class_idx], axis=1)
            bar = ax.bar(range(num_bins_y), class_counts_y[:, class_idx], bottom=bottom, label=f'Class {class_idx}',
                color=f'C{class_idx}', alpha=0.7)
            if class_idx == 1:
                bar.set_label("Walls")
            elif class_idx == 9:
                bar.set_label("Objects")
            else:
                bar.set_label("__Other")
            ax.tick_params(axis='both', which='major', labelsize=font_size)
            # ax.set_xticks(range(num_bins_y))
            # ax.set_xticklabels([f'{i+y_min}' for i in range(num_bins_y)])
            ax.set_title('Y-axis', weight='bold',fontsize=font_size)
            ax.legend(fontsize=font_size, loc='upper left')

        # plt.legend( )
        plt.tight_layout()
        plt.show()
        
    # Function to extract the principal axis of the point cloud
    def extract_principal_axis(self, ptc:o3d.geometry.PointCloud):
        # Project points onto XY plane
        np_points = np.asarray(ptc.points)
        # np_points = np_points[:, :2]
        
        # Check if the pointcloud has the correct shape
        if np_points.ndim != 2 or np_points.shape[1] != 3:
            raise ValueError("Input pointcloud must be an Nx3 numpy array.")

        # Perform PCA on the pointcloud
        pca = PCA(n_components=3)
        pca.fit(np_points)

        # Get the principal components (eigenvectors)
        components = pca.components_

        principal_frame = o3d.geometry.LineSet()
        points = np.zeros((4, 3))
        points[1] = components[0]
        points[2] = components[1]
        points[3] = components[2]
        principal_frame.points = o3d.utility.Vector3dVector(points)
        principal_frame.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])

        # self.geometries["Principal Frame"] = principal_frame
        # self.geometry_names.append("Principal Frame")

        # Create a rotation matrix from the principal components
        rotation_matrix = np.linalg.inv(components)
        # full_rotation_matrix = np.eye(3)
        # full_rotation_matrix[:2, :2] = rotation_matrix
        return rotation_matrix
    
    # Function to create a bounding box aligned only with z axis of the input point cloud
    def z_aligned_bounding_box(self, ptc:o3d.geometry.PointCloud) -> o3d.geometry.LineSet:
        # Project points onto XY plane
        np_points = np.asarray(ptc.points)
        max_z = np.max( np_points[:, 2] ) 
        min_z = np.min( np_points[:, 2] ) 

        projected_points = np_points[:, :2].T

        # calculate the means and the covariance matrix
        means = np.mean(projected_points, axis=1) 
        cov = np.cov(projected_points)
        # Calculate the eigen values and eigen vectors of the covariance matrix
        _, evec = np.linalg.eig(cov)

        centered_points = projected_points - means[:, np.newaxis]
        v = evec[:, 0]
        theta_pcl = np.arctan(v[1]/v[0])

        rot = lambda theta: np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],
                              [np.sin(np.deg2rad(theta)),  np.cos(np.deg2rad(theta))]])

        aligned_coords = np.matmul(rot(np.rad2deg(-theta_pcl)), centered_points)
        xmin, xmax, ymin, ymax = np.min(aligned_coords[0,:]), np.max(aligned_coords[0,:]), np.min(aligned_coords[1,:]), np.max(aligned_coords[1,:])

        rectangle_coords = np.array([[xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax]])

        # Rotate the rectangle back to the original orientation
        aligned_coords = np.matmul(rot(np.rad2deg(theta_pcl)), rectangle_coords)
        aligned_coords += means[:, np.newaxis]

        # Extrude the rectangle to form a 3D bounding box
        bb = np.zeros((8, 3))
        bb[:4, :2] = aligned_coords.T
        bb[4:, :2] = aligned_coords.T
        bb[4:, 2] = max_z
        bb[:4, 2] = min_z

        # Create the Open3D bounding box
        o3d_bb = o3d.geometry.LineSet()
        o3d_bb.points = o3d.utility.Vector3dVector(bb)
        lines = [[0, 1], [1, 2], [2, 3], [3, 0],
                 [4, 5], [5, 6], [6, 7], [7, 4],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        o3d_bb.lines = o3d.utility.Vector2iVector(lines)


        return o3d_bb

    # A function to filter the ptc by class color 
    # @param voxel: bool - if True, the function will create a voxel grid of the object clusters
    def filter_point_clouds(self, voxel=False, voxel_size=0.3):
        if self.mapper.point_classes is None or self.vis is False:
            raise ValueError("No point classes found or visualization is disabled")
        if not self.mapper.viewer.updated:
            return
        class_idx = self.mapper.viewer.class_idx
        class_name = self.mapper.class_labels[class_idx]

        walls_idx = 1       # constant class index 
        walls_indices = np.where(self.mapper.point_classes == walls_idx)[0]

        class_indices, labels = self.cluster_class(class_idx, 0.3, 8)

        class_entries = class_indices.shape[0] 
        if (class_entries <= 2):
            print("Skipping ", class_name)
            self.mapper.viewer.class_switch_cb()
            return

        for cluster in np.unique(labels):
            cluster_indices = np.where(labels == cluster)[0]
            cluster_ptc = self.mapper.raw_point_cloud.select_by_index(class_indices[cluster_indices], invert=False)
            aa_bb = cluster_ptc.get_axis_aligned_bounding_box()
            # aa_bb = cluster_ptc.get_oriented_bounding_box()
            # aa_bb = self.part_aligned_bounding_box(cluster_ptc)
            # aa_bb.paint_uniform_color(self.dataset[class_name]/255)
            aa_bb.color = self.mapper.dataset[class_name]/255

            bb_name = f"{class_name}{cluster}"
            
            if voxel:
                voxel_bb = o3d.geometry.VoxelGrid.create_dense(aa_bb.get_center()-aa_bb.get_half_extent(),aa_bb.color, voxel_size, *aa_bb.get_extent())
                self.mapper.viewer.add_geometry(bb_name, voxel_bb)
            else:   
                con_hull = cluster_ptc.compute_convex_hull()[0]
                con_hull.paint_uniform_color(self.mapper.dataset[class_name]/255)
                self.mapper.viewer.add_geometry(bb_name, con_hull)

        
        if self.mapper.viewer.updated:
            print("Picked points: ", class_entries)
            print("Displaying class: ", class_name)
            self.mapper.viewer.updated = False
            
        # class + walls
        display_indices = np.unique(np.concatenate((class_indices, walls_indices), axis=0))
        self.mapper.viewer.raw_ptc = self.mapper.raw_point_cloud.select_by_index(display_indices, invert=False)
        # if voxel:
        #     voxel_map = o3d.geometry.VoxelGrid.create_from_point_cloud(self.mapper.viewer.raw_ptc, voxel_size)
        #     self.mapper.viewer.update_geometry(self.mapper.viewer.raw_ptc_name, voxel_map)
        #     # self.mapper.viewer.raw_ptc = voxel_map
        # else:
        # self.mapper.viewer.raw_ptc = self.mapper.viewer.raw_ptc


    # Cluster all classes separetely and fuse the results
    # NOT TESTED
    # TODO: remove? 
    def cluster_all_classes(self):
        full_class_indices = []
        inital_len = len(self.mapper.rgb_load)
        # Run DBSCAN clustering algorithm separetely for each class to remove noise
        for class_idx in range(1, len(self.dataset)):
            # Skip the ceiling and floor classes TODO: lights too I guess
            if class_idx == 4 or class_idx == 6:
                continue
            # Get the indices of the points belonging to the current class
            class_indices, labels = self.cluster_class(class_idx)
            if class_indices.shape[0] < 4:
                continue
            # Save the class indicies
            full_class_indices.append(class_indices)

        full_class_indices = np.concatenate(full_class_indices, axis=0)
        self.raw_point_cloud = self.mapper.raw_point_cloud.select_by_index(full_class_indices, invert=False)
        self.mapper.rgb_load = self.mapper.rgb_load[full_class_indices]
        self.build_point_classes()

        print(f"Removed {inital_len - len(self.mapper.rgb_load)} points")

    # Cluster a particular class using DBSCAN. Based on self.raw_point_cloud
    def cluster_class(self, class_idx:int, eps:float = 0.5, min_points:int = 8) -> tuple:
        if self.mapper.point_classes is None:
            raise ValueError("No point classes found")
        
        class_indices = np.where(self.mapper.point_classes == class_idx)[0]
        class_points = self.mapper.raw_point_cloud.select_by_index(class_indices, invert=False)

        # Convert the Open3D point cloud to a numpy array
        class_points_np = np.asarray(class_points.points)
        if class_points_np.shape[0] < 10:
            return np.empty((0, 1)), np.empty((0, 1))

        # Run DBSCAN clustering algorithm
        labels = np.array(class_points.cluster_dbscan(eps, min_points, print_progress=False))
        # print(sum(labels!=-1))

        class_indices = class_indices[labels != -1]
        labels = labels[labels != -1]
       
        return class_indices, labels
    
    def test_esdf_voxel_grid(self, voxel_size = 0.2, filters=False):
        if filters:
            wall_points = self.merge_wall_classes([1, 9, 15, 23], obj_outlier_removal=True)
        else:
            wall_points = self.mapper.raw_point_cloud

        min_bound = self.mapper.shared_min_bound
        max_bound = self.mapper.shared_max_bound 
        voxel_grid = np.zeros((int((max_bound[0] - min_bound[0]) / voxel_size)+1,
                            int((max_bound[1] - min_bound[1]) / voxel_size)+1,
                            int((max_bound[2] - min_bound[2]) / voxel_size)+1), dtype=bool)
        
        # Fill the voxel grid
        np_points = np.asarray(wall_points.points)
        for point in np_points:
            x = int((point[0] - min_bound[0]) / voxel_size)
            y = int((point[1] - min_bound[1]) / voxel_size)
            z = int((point[2] - min_bound[2]) / voxel_size)
            voxel_grid[x, y, z] = True
        
        
        esdf = self.compute_esdf(voxel_grid) * voxel_size

        valid_slice = esdf[:, :, 9:32]  # HACK 

        # esdf[esdf>0.2] = 0
        
        # Function to return the element with the maximum absolute value but keeps the sign
        def max_abs_with_sign(x):
            return max(x, key=abs)

        # Get abs max in each stack    
        # flat_esdf = np.average(valid_slice, axis=2)
        # flat_esdf = np.min(valid_slice, axis=2)
        # Apply the function to each stack in the 3D array
        flat_esdf = np.apply_along_axis(max_abs_with_sign, 2, valid_slice)
        flat_esdf[flat_esdf > 3] = -2
        
        # Preview a slice of the voxel grid
        plt.imshow(flat_esdf)
        plt.colorbar()
        plt.show()
        
        return flat_esdf
        
            
        

    def compute_esdf(self, voxel_map):
        """
        Compute the Euclidean Signed Distance Field for a 3D voxel map.

        Parameters:
        - voxel_map (np.ndarray): A 3D numpy array where `1` indicates occupied voxel and `0` indicates free space.

        Returns:
        - np.ndarray: A 3D array of the same size with distances to the nearest occupied voxel.
        """
        # Ensure voxel_map is a numpy array with binary data
        assert isinstance(voxel_map, np.ndarray), "Input must be a numpy array"
        assert voxel_map.ndim == 3, "Input must be a 3D array"

        # Invert the voxel map for distance transform: occupied (1) becomes False, free (0) becomes True
        inverted_voxel_map = (voxel_map == 0)

        # Compute the Euclidean distance transform
        distance_field = distance_transform_edt(inverted_voxel_map)
        distance_field[voxel_map == 1] = -10 # TODO: replace with  a proper inverted distance map 

        return distance_field

    
    # Function to merge rooms in an overpartitioned voxel grid
    def merge_partitions(self, voxel_grid, ptc, use_trajectory=False) -> np.ndarray:
        if use_trajectory and self.trajectories is None:
            print("No trajectories found")
            return voxel_grid
        
        # Walls
        z_mid = voxel_grid.shape[2] // 2
        x_walls = np.where(voxel_grid[:,0,z_mid] == 1)[0]
        y_walls = np.where(voxel_grid[0,:,z_mid] == 1)[0]
        
        # Cut things outside the outer walls
        voxel_grid[:x_walls[0]-1, :, :] = 0
        voxel_grid[x_walls[-1]+1:, :, :] = 0
        voxel_grid[:, :y_walls[0]-1, :] = 0
        voxel_grid[:, y_walls[-1]+1:, :] = 0
        
        ptc_min_bound = ptc.get_min_bound()
        ptc_max_bound = ptc.get_max_bound()
        voxel_size = 0.2
        wall_width = 0.6 # The width of the bb used to check for inliers
        wall_thresh = 5
        prop_thresh = 0.7
        threshold = 20
        
        # traverse the walls and merge the partitions
        # TODO: unify the code for x and y walls
        for x in x_walls:
            for y_ind in range(1, len(y_walls)):
                prev_y = y_walls[y_ind-1]
                y = y_walls[y_ind]
                wall_length = y - prev_y # in voxels
                
                bb_min_bound = np.array([x*voxel_size + ptc_min_bound[0] - wall_width/2,
                                         prev_y*voxel_size + ptc_min_bound[1], 
                                         ptc_min_bound[2]]) 
                bb_max_bound = np.array([(x+0)*voxel_size + ptc_min_bound[0] + wall_width/2,
                                         y*voxel_size + ptc_min_bound[1], 
                                         ptc_max_bound[2]])
                
                wall_bb = o3d.geometry.AxisAlignedBoundingBox(bb_min_bound, bb_max_bound)

                # Remove wall partitions intersecting with the trajectory
                if use_trajectory and self.trajectories is not None:
                    traj_ptc = self.trajectories.crop(wall_bb)
                    traj_points = np.asarray(traj_ptc.points)
                    n_traj_points = traj_points.shape[0]
                    print(n_traj_points)
                    if n_traj_points > 0:
                        voxel_grid[x, prev_y:y, :] = 0
                        continue

                wall_ptc = ptc.crop(wall_bb)
                wall_points = np.asarray(wall_ptc.points)
                n_wall_points = wall_points.shape[0]
                
                if n_wall_points < threshold:
                    voxel_grid[x, prev_y:y, :] = 0
                    continue
                
                wall_hist, wall_bins = np.histogram(wall_points[:, 1], bins=wall_length)
                # Get the percentage of bins with less than wall_thresh points
                # plt.bar(wall_bins[:-1], wall_hist, width=wall_bins[1] - wall_bins[0], align='edge', edgecolor='black')
                # plt.show()
                proportion = np.sum(wall_hist < wall_thresh) / wall_length
                # print(proportion)
                if proportion > prop_thresh:
                    voxel_grid[x, prev_y:y, :] = 0
                    continue
                     
        for y in y_walls:
            for x_ind in range(1, len(x_walls)):
                prev_x = x_walls[x_ind-1]
                x = x_walls[x_ind]
                wall_length = x - prev_x # in voxels
                
                bb_min_bound = np.array([prev_x*voxel_size + ptc_min_bound[0], 
                                         y*voxel_size + ptc_min_bound[1] - wall_width/2, 
                                         ptc_min_bound[2]])
                bb_max_bound = np.array([x*voxel_size + ptc_min_bound[0], 
                                         (y+0)*voxel_size + ptc_min_bound[1] + wall_width/2, 
                                         ptc_max_bound[2]])
                
                wall_bb = o3d.geometry.AxisAlignedBoundingBox(bb_min_bound, bb_max_bound)

                if use_trajectory and self.trajectories is not None:
                    traj_ptc = self.trajectories.crop(wall_bb)
                    traj_points = np.asarray(traj_ptc.points)
                    n_traj_points = traj_points.shape[0]
                    print(n_traj_points)
                    if n_traj_points > 0:
                        voxel_grid[prev_x:x, y, :] = 0
                        continue

                wall_ptc = ptc.crop(wall_bb)
                
                wall_points = np.asarray(wall_ptc.points)
                n_wall_points = wall_points.shape[0]
                if n_wall_points < threshold:
                    voxel_grid[prev_x:x, y, :] = 0
                    continue
                
                wall_hist, wall_bins = np.histogram(wall_points[:, 0], bins=wall_length)
                # Get the percentage of bins with less than wall_thresh points
                proportion = np.sum(wall_hist < wall_thresh) / wall_length
                # print(proportion)
                # plt.bar(wall_bins[:-1], wall_hist, width=wall_bins[1] - wall_bins[0], align='edge', edgecolor='black')
                # plt.show()
                if proportion > prop_thresh:
                    voxel_grid[x, prev_y:y, :] = 0
                    continue
                
        return voxel_grid
                         
    # A simple function to convert self.raw_point_cloud to a voxel_grid
    def create_o3d_voxel_grid(self, voxel_size=0.3):
        o3d_vox_ptc = self.mapper.raw_point_cloud
        o3d_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_vox_ptc, voxel_size)

        self.mapper.viewer.update_geometry(self.mapper.point_cloud_o3d_name, o3d_vox)
    
    # A floorplan extraction pipeline 
    def rebuild_rooms(self):
        voxel_size = 0.3
        wall_points = self.merge_wall_classes([1, 9, 15, 19, 23, 101, 145,  4, 6, 83], obj_outlier_removal=True)
        
        # structures: 19806+679+1264+102+840+1582+519=24672
        # horizontals: 2433+982+109=3524
        # 24692+3524=28316

        # 18007 / 28316 = 0.635
        # 24672*0.635 = 15727
        # 3524*0.635 = 2280

        # o3d.visualization.draw_geometries([wall_points],  point_show_normal=True)
        
        # Create a voxel grid
        max_bound = self.mapper.raw_point_cloud.get_max_bound()
        min_bound = self.mapper.raw_point_cloud.get_min_bound()
        voxel_grid = np.zeros((int((max_bound[0] - min_bound[0]) / voxel_size)+1,
                        int((max_bound[1] - min_bound[1]) / voxel_size)+1,
                        int((max_bound[2] - min_bound[2]) / voxel_size)+1), dtype=int)

        wall_voxels = self.fit_vert_planes(wall_points, voxel_grid, voxel_size, False)
        # wall_voxels = self.fit_vert_planes_comb(wall_points, voxel_grid, voxel_size, False)
        # Preview a slice of the voxel grid
        # plt.imshow(wall_voxels[:, :, 10])
        # plt.show()

        floor_wall_voxels = self.fit_hor_planes(wall_voxels, voxel_size, classes=[4,]) # 6, 83
        # Preview a slice of the voxel grid
        plt.imshow(floor_wall_voxels[:, :, 5])
        plt.show()

        populated_voxel_grid = self.draw_voxel_grid(floor_wall_voxels, voxel_size)

        # cutter = RMFCut()
        # cut = cutter.segment_3d_grid(populated_voxel_grid, voxel_size) 
        # pf = cutter.build_pf(cut, populated_voxel_grid)
        
        # # Take z-axis max
        # max_pf = np.max(pf[:,:,:10], axis=2)
        # # Vizualize a slice of the pf
        # plt.imshow(max_pf.T)
        # plt.show()
    
    # Extract peaks of a histogram with several convolutions 
    def get_hist_peaks(self, hist:np.ndarray, threshold:float):
        # Smooth the histograms
        hist = np.convolve(hist, np.ones(3) / 3, mode='same')
        # Sharpen the histograms
        kernel = np.zeros(3)
        kernel[1] = 3
        kernel -= np.ones(3) / 3
        # Find valleys
        # kernel = self.get_kernel(7, 3)
        hist = np.convolve(hist, kernel, mode='same')

        # _|-|_ - kernel
        kernel = -1/7*np.ones(7)
        kernel[2:5] = 1
        hist = np.convolve(hist, kernel, mode='same')

        # Find local maxima in the smoothed histograms
        peaks = np.where((hist[1:-1] - hist[:-2] > threshold) & (hist[1:-1] - hist[2:] > threshold))[0] + 1
  
        return peaks, hist
    
    # Extract peaks of a histogram visible on several scales 
    def detect_walls(self, signal:np.ndarray) -> tuple:
        bin_counts = range(1, 10)*(max(signal) - min(signal)) # assuming meters 
        all_peaks = []
        for bin_count in bin_counts:
            hist, bins = np.histogram(signal, bins=int(bin_count) )
            hist = np.convolve(hist, np.ones(3)/3, mode="same")
            # All bins higher than their neighbours
            peaks = np.where((hist[1:-1] - hist[:-2] > 3) & 
                             (hist[1:-1] - hist[2:]  > 3 ))[0] + 1
            for peak in peaks:
                all_peaks.append( (int(bin_count), bins[peak]) )
        
        all_peaks = np.array(all_peaks)
        # plt.figure()
        # plt.scatter(all_peaks[:, 0], all_peaks[:, 1])
        all_hist, all_bins = np.histogram(all_peaks[:,1], bins=int(bin_count) )     # TODO: consider bin width
        mean_detection = np.mean(all_hist[all_hist > 0]) + 1
        max_peaks = np.where(all_hist >= mean_detection)[0] + 1
        
        # plt.bar(all_bins[:-1], all_hist, width=all_bins[1] - all_bins[0], align='edge', edgecolor='black' )
        # plt.hlines(mean_detection, all_bins[0], all_bins[-1], color='red')
        # plt.show()
        
        return max_peaks, all_bins
        
       
    # Fit  plane to the points based on their distributions 
    # Assuming the ptc is already rotated
    def fit_vert_planes(self, ptc:o3d.geometry.PointCloud, voxel_grid=None, voxel_size=0.3, partition_mode = True):
        np_points = np.asarray(ptc.points)
        eps = 0.1
        min_points = 30
        wall_threshold = 5

        min_bound = self.mapper.raw_point_cloud.get_min_bound()
        max_bound = self.mapper.raw_point_cloud.get_max_bound()
        # Create a voxel grid
        if voxel_grid is None:
            min_bound = ptc.get_min_bound()
            max_bound = ptc.get_max_bound()
            voxel_grid = np.zeros((int((max_bound[0] - min_bound[0]) / voxel_size)+1,
                                int((max_bound[1] - min_bound[1]) / voxel_size)+1,
                                int((max_bound[2] - min_bound[2]) / voxel_size)+1), dtype=bool)    

        # Extract x, y, and z coordinates from the pointcloud
        x = np_points[:, 0]
        y = np_points[:, 1]

        # Compute the distribution of points along the x-axis
        # x_hist, x_bins = np.histogram(x, bins=bins)
        # x_peaks, x_hist = self.get_hist_peaks(x_hist, 6)
        x_peaks, x_bins = self.detect_walls(x)
        
        # y_hist, y_bins = np.histogram(y, bins=bins)
        # y_peaks, y_hist = self.get_hist_peaks(y_hist, 6)
        y_peaks, y_bins = self.detect_walls(y)
        
        # Preview both histograms with the peaks
        # plt.figure()
        # # Subplot 1
        # plt.subplot(1, 2, 1)
        # plt.bar(x_bins[:-1], x_hist, width=x_bins[1] - x_bins[0], align='edge', edgecolor='black')
        # plt.bar(x_bins[x_peaks], x_hist[x_peaks], color='red', width=x_bins[1] - x_bins[0], align='edge', edgecolor='black')
        # # Subplot 2
        # plt.subplot(1, 2, 2)
        # plt.bar(y_bins[:-1], y_hist, width=y_bins[1] - y_bins[0], align='edge', edgecolor='black')
        # plt.bar(y_bins[y_peaks], y_hist[y_peaks], color='red', width=y_bins[1] - y_bins[0], align='edge', edgecolor='black')
        # plt.legend()
        # plt.show()  
        
        plt.figure()
        plt.scatter(x, y, marker='.')
        for x_p in x_peaks:
            plt.axvline(x_bins[x_p], color="red")
        for y_p in y_peaks:
            plt.axhline(y_bins[y_p], color="red")
        plt.axis('equal')
        plt.show()

        # Iterate over the peaks found in the x-axis distribution
        for peak_idx in x_peaks:
            # Find the corresponding points in the pointcloud
            x_min = x_bins[peak_idx] - eps
            x_max = x_min + (x_bins[1]-x_bins[0]) + eps 
            peak_points = np_points[(x >= x_min) & (x < x_max)]
            
            # Compute the bounds of the peak points along other axes
            y_min, y_max = np.min(peak_points[:, 1]), np.max(peak_points[:, 1])
            z_min, z_max = np.min(peak_points[:, 2]), np.max(peak_points[:, 2])
            wall_length = int(y_max - y_min)
            # print(" Height: ", z_max - z_min)
            
            if partition_mode:
                voxel_grid[int((x_max - min_bound[0]) / voxel_size), #:int((x_max - min_bound[0]) / voxel_size)
                            :,
                            int((z_min - min_bound[2]) / voxel_size):int((z_max - min_bound[2]) / voxel_size)] = 1
                continue

            if peak_points.shape[0] < min_points:
                continue

            xy_hist, xy_bins = np.histogram(peak_points[:, 1], bins=wall_length*2)
            xy_hist = np.convolve(xy_hist, np.ones(3) / 3, mode='same')
            # Find voids in the other dimension
            xy_peaks = np.where(xy_hist > wall_threshold)[0]
            # # Check where peak_points are in np_points vizually

            # Create a figure with gridspec
            plt.figure()
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

            # First subplot
            ax1 = plt.subplot(gs[0])
            plt.scatter(np_points[:, 0], np_points[:, 1], color='blue', marker='.')
            plt.scatter(peak_points[:, 0], peak_points[:, 1], color='red', marker='.')
            plt.axis('equal')
            # Second subplot, sharing y-axis with the first subplot
            ax2 = plt.subplot(gs[1], sharey=ax1)
            plt.barh(xy_bins[:-1], width=xy_hist, height=xy_bins[1] - xy_bins[0], align='edge', edgecolor='black')
            plt.vlines(wall_threshold, xy_bins[0], xy_bins[-1], "red", lw=4)

            # Show the plot
            plt.show()

            # plt.figure()
            # ax1 = plt.subplot(1, 2, 1)
            # plt.scatter(np_points[:, 0], np_points[:, 1], color='blue',  marker='.')
            # plt.scatter(peak_points[:, 0], peak_points[:, 1], color='red',  marker='.')
            # # plt.axis('equal')
            # plt.subplot(1, 2, 2, sharey=ax1)
            # plt.barh(xy_bins[:-1], width=xy_hist, height=xy_bins[1] - xy_bins[0], align='edge', edgecolor='black')  
            # plt.vlines(wall_threshold, xy_bins[0], xy_bins[-1], "red", lw=4)
            # plt.show()


            # There is probably a smarter way to do this\
            for wall_idx in xy_peaks:
                y_min = xy_bins[wall_idx]
                y_max = xy_bins[wall_idx + 1]
                
                voxel_grid[int((x_min - min_bound[0]) / voxel_size),
                            int((y_min - min_bound[1]) / voxel_size):int((y_max - min_bound[1]) / voxel_size),
                            int((z_min - min_bound[2]) / voxel_size):int((z_max - min_bound[2]) / voxel_size)] = 1
      
        # Iterate over the peaks found in the y-axis distribution
        for peak_idx in y_peaks:
            # Find the corresponding points in the pointcloud
            y_min = y_bins[peak_idx] - eps
            y_max = y_min + (y_bins[1]-y_bins[0]) + eps 
            peak_points = np_points[(y >= y_min) & (y < y_max)]
            # Compute the bounds of the peak points along other axes
            x_min, x_max = np.min(peak_points[:, 0]), np.max(peak_points[:, 0])
            z_min, z_max = np.min(peak_points[:, 2]), np.max(peak_points[:, 2])
            wall_length = int(x_max - x_min)
            
            if partition_mode:
                voxel_grid[:,
                    int((y_max - min_bound[1]) / voxel_size), #:int((y_max - min_bound[1]) / voxel_size)
                    int((z_min - min_bound[2]) / voxel_size):int((z_max - min_bound[2]) / voxel_size)] = 1
                            
                continue            

            if peak_points.shape[0] < min_points:
                continue

            yx_hist, yx_bins = np.histogram(peak_points[:, 0], bins=wall_length*2)
            yx_hist = np.convolve(yx_hist, np.ones(3) / 3, mode='same')
            # Find voids in the other dimension
            yx_peaks = np.where(yx_hist > wall_threshold)[0]

            # plt.figure()
            # ax1 = plt.subplot(2, 1, 1)
            # plt.scatter(np_points[:, 0], np_points[:, 1], color='blue')
            # plt.scatter(peak_points[:, 0], peak_points[:, 1], color='red')
            # plt.subplot(2, 1, 2, sharex=ax1)
            # plt.bar(yx_bins[:-1], yx_hist, width=yx_bins[1] - yx_bins[0], align='edge', edgecolor='black')  
            # plt.show()

            # There is probably a smarter way to do this\
            for wall_idx in yx_peaks:
                x_min = yx_bins[wall_idx]
                x_max = x_min + yx_bins[1] - yx_bins[0]
                
                voxel_grid[int((x_min - min_bound[0]) / voxel_size):int((x_max - min_bound[0]) / voxel_size),
                            int((y_min - min_bound[1]) / voxel_size),
                            int((z_min - min_bound[2]) / voxel_size):int((z_max - min_bound[2]) / voxel_size)] = 1
        
        if partition_mode:
            voxel_grid = self.merge_partitions(voxel_grid, ptc, use_trajectory=False)
        
        return voxel_grid
    
    # Fit plane to the vertical surfaces locally 
    # Assuming the ptc is already rotated
    def fit_vert_planes_comb(self, ptc:o3d.geometry.PointCloud, voxel_grid=None, voxel_size=0.3, partition_mode=True):
        np_points = np.asarray(ptc.points)

        min_bound = self.mapper.raw_point_cloud.get_min_bound()
        max_bound = self.mapper.raw_point_cloud.get_max_bound()
        # Create a voxel grid
        if voxel_grid is None:
            min_bound = ptc.get_min_bound()
            max_bound = ptc.get_max_bound()
            voxel_grid = np.zeros((int((max_bound[0] - min_bound[0]) / voxel_size)+1,
                                int((max_bound[1] - min_bound[1]) / voxel_size)+1,
                                int((max_bound[2] - min_bound[2]) / voxel_size)+1), dtype=bool)    

        # Extract x, y, and z coordinates from the pointcloud
        x = np_points[:, 0]
        y = np_points[:, 1]
        z = np_points[:, 2]

        # Compute the distribution of points along the x-axis
        # x_hist, x_bins = np.histogram(x, bins=bins)
        # x_peaks, x_hist = self.get_hist_peaks(x_hist, 6)
        
        # y_hist, y_bins = np.histogram(y, bins=bins)
        # y_peaks, y_hist = self.get_hist_peaks(y_hist, 6)
        
        x_peaks, x_bins = self.detect_walls(x)
        y_peaks, y_bins = self.detect_walls(y)
        
        # Preview both histograms with the peaks
        # plt.figure()
        # # Subplot 1
        # plt.subplot(1, 2, 1)
        # plt.bar(x_bins[:-1], x_hist, width=x_bins[1] - x_bins[0], align='edge', edgecolor='black')
        # plt.bar(x_bins[x_peaks], x_hist[x_peaks], color='red', width=x_bins[1] - x_bins[0], align='edge', edgecolor='black')
        # # Subplot 2
        # plt.subplot(1, 2, 2)
        # plt.bar(y_bins[:-1], y_hist, width=y_bins[1] - y_bins[0], align='edge', edgecolor='black')
        # plt.bar(y_bins[y_peaks], y_hist[y_peaks], color='red', width=y_bins[1] - y_bins[0], align='edge', edgecolor='black')
        # plt.legend()
        # plt.show()  

        # Iterate over the peaks found in the x-axis distribution
        for n in range(1, len(x_peaks)):
            peak_idx = x_peaks[n]
            prev_peak_idx = x_peaks[n-1]
            # Isolate the partition 
            x_min = x_bins[prev_peak_idx]
            x_max = x_bins[peak_idx]
            # Find the corresponding points in the pointcloud
            partition_points = np_points[(x >= x_min) & (x < x_max)]
            
            # Compute the bounds of the peak points along other axes
            y_min, y_max = np.min(partition_points[:, 1]), np.max(partition_points[:, 1])
            z_min, z_max = np.min(partition_points[:, 2]), np.max(partition_points[:, 2])
            part_len = y_max - y_min
            
            # Find the peaks in the y-axis distribution
            # xy_hist, xy_bins = np.histogram(partition_points[:, 1], bins=int(part_len/voxel_size))
            # xy_peaks, xy_hist = self.get_hist_peaks(xy_hist, 6)
            xy_peaks, xy_bins = self.detect_walls(partition_points[:, 1])
            # # Check where peak_points are in np_points vizually
            # plt.figure()
            # ax1 = plt.subplot(1, 2, 1)
            # plt.scatter(np_points[:, 0], np_points[:, 1], color='blue')
            # plt.scatter(partition_points[:, 0], partition_points[:, 1], color='red')
            # # plt.scatter(partition_points[xy_peaks, 0], partition_points[xy_peaks, 1], color='green')
            # plt.subplot(1, 2, 2, sharey=ax1)
            # plt.barh(xy_bins[:-1], width=xy_hist, height=xy_bins[1] - xy_bins[0], align='edge', edgecolor='black')  
            # plt.barh(xy_bins[xy_peaks], width=xy_hist[xy_peaks], height=xy_bins[1] - xy_bins[0], color='green', align='edge', edgecolor='black')
            # plt.show()

            # Draw the long x-axis walls
            # voxel_grid[int((x_min - min_bound[0]) / voxel_size),
            #             int((y_min - min_bound[1]) / voxel_size):int((y_max - min_bound[1]) / voxel_size),
            #             int((z_min - min_bound[2]) / voxel_size):int((z_max - min_bound[2]) / voxel_size)] = 1
            # voxel_grid[int((x_max - min_bound[0]) / voxel_size),
            #             int((y_min - min_bound[1]) / voxel_size):int((y_max - min_bound[1]) / voxel_size),
            #             int((z_min - min_bound[2]) / voxel_size):int((z_max - min_bound[2]) / voxel_size)] = 1

            # There is probably a smarter way to do this\
            for wall_idx in xy_peaks:
                y_min = xy_bins[wall_idx]
                
                voxel_grid[int((x_min - min_bound[0]) / voxel_size):int((x_max - min_bound[0]) / voxel_size),
                           int((y_min - min_bound[1]) / voxel_size),
                           int((z_min - min_bound[2]) / voxel_size):int((z_max - min_bound[2]) / voxel_size)] = 1
        
        for n in range(1, len(y_peaks)):
            peak_idx = y_peaks[n]
            prev_peak_idx = y_peaks[n-1]
            # Isolate the partition 
            y_min = y_bins[prev_peak_idx]
            y_max = y_bins[peak_idx]
            # Find the corresponding points in the pointcloud
            partition_points = np_points[(y >= y_min) & (y < y_max)]
            
            # Compute the bounds of the peak points along other axes
            x_min, x_max = np.min(partition_points[:, 0]), np.max(partition_points[:, 0])
            z_min, z_max = np.min(partition_points[:, 2]), np.max(partition_points[:, 2])
            part_len = x_max - x_min
            
            # Find the peaks in the x-axis distribution
            # yx_hist, yx_bins = np.histogram(partition_points[:, 0], bins=int(part_len/voxel_size))
            # yx_peaks, yx_hist = self.get_hist_peaks(yx_hist, 6)
            yx_peaks, yx_bins = self.detect_walls(partition_points[:, 0])
            # # Check where peak_points are in np_points vizually
            # plt.figure()
            # ax1 = plt.subplot(1, 2, 1)
            # plt.scatter(np_points[:, 0], np_points[:, 1], color='blue')
            # plt.scatter(partition_points[:, 0], partition_points[:, 1], color='red')
            # # plt.scatter(partition_points[xy_peaks, 0], partition_points[xy_peaks, 1], color='green')
            # plt.subplot(1, 2, 2, sharey=ax1)
            # plt.barh(yx_bins[:-1], width=yx_hist, height=yx_bins[1] - yx_bins[0], align='edge', edgecolor='black')  
            # plt.barh(yx_bins[yx_peaks], width=yx_hist[yx_peaks], height=yx_bins[1] - yx_bins[0], color='green', align='edge', edgecolor='black')
            # plt.show()

            # Draw the long x-axis walls
            # voxel_grid[int((x_min - min_bound[0]) / voxel_size),
            #             int((y_min - min_bound[1]) / voxel_size):int((y_max - min_bound[1]) / voxel_size),
            #             int((z_min - min_bound[2]) / voxel_size):int((z_max - min_bound[2]) / voxel_size)] = 1
            # voxel_grid[int((x_max - min_bound[0]) / voxel_size),
            #             int((y_min - min_bound[1]) / voxel_size):int((y_max - min_bound[1]) / voxel_size),
            #             int((z_min - min_bound[2]) / voxel_size):int((z_max - min_bound[2]) / voxel_size)] = 1

            # There is probably a smarter way to do this\
            for wall_idx in yx_peaks:
                x_min = yx_bins[wall_idx]
                
                voxel_grid[int((x_min - min_bound[0]) / voxel_size),
                           int((y_min - min_bound[1]) / voxel_size):int((y_max - min_bound[1]) / voxel_size),
                           int((z_min - min_bound[2]) / voxel_size):int((z_max - min_bound[2]) / voxel_size)] = 1

        return voxel_grid    

    #    __    __
    # __| |___| |__
    # 1 c gap c 1 = size
    def get_kernel(self, size, gap):
        c = (size - 2 - gap) // 2 + 1
        kernel = np.zeros(size)
        kernel[1:c] = 0.5*c
        kernel[-c:-1] = 0.5*c

        return kernel

    # Fit horizontal planes to the points based on their distributions
    # Relies on self.raw_point_cloud
    def fit_hor_planes(self, voxel_grid, voxel_size=0.3, classes=[4, 6, 83]):
        pcd = self.mapper.raw_point_cloud

        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        
        for plane_class_id in classes: 
            class_indices, labels = self.cluster_class(plane_class_id, eps=0.8, min_points=8)
            for cluster in np.unique(labels):
                cluster_indices = np.where(labels == cluster)[0]
                cluster_ptc  = self.mapper.raw_point_cloud.select_by_index(class_indices[cluster_indices], invert=False)
                aa_bb = cluster_ptc.get_axis_aligned_bounding_box()
                # Find the voxel coordinates of the bounding box edges
                bb_center = aa_bb.get_center()
                bb_extent = aa_bb.get_half_extent()
                bb_min = bb_center - bb_extent
                bb_max = bb_center + bb_extent
                # Fill the voxel grid with the bounding box
                voxel_min = ((bb_min- min_bound ) / voxel_size).astype(int)
                voxel_max = ((bb_max- min_bound ) / voxel_size).astype(int)
                voxel_mean = ((bb_center - min_bound) / voxel_size).astype(int)
                mean_vox_z = voxel_mean[2]+1 if plane_class_id==4 else voxel_max[2] 

                voxel_grid[voxel_min[0]:voxel_max[0], voxel_min[1]:voxel_max[1], mean_vox_z-1] = plane_class_id

        # for i, point in enumerate(pcd.points):
        #     voxel_coord = ((point - min_bound) / voxel_size).astype(int)
        #     if self.mapper.point_classes[i] != 1:
        #         continue
        #     voxel_grid[voxel_coord[0], voxel_coord[1], voxel_coord[2]] = True
            
        return  voxel_grid

    # Populates the voxel grid with the objects 
    def draw_voxel_grid(self, voxel_grid:np.ndarray, voxel_size=0.3, classes_to_hide = [1, 4, 6, 83]):
        # empty TriangleMesh object that will contain the cubes
        vox_mesh = o3d.geometry.TriangleMesh()

        min_bound = self.mapper.raw_point_cloud.get_min_bound()

        populate_grid = False
        if populate_grid:
            # Iterate over all classes and draw their bounding boxes
            for class_id in range(2, len(self.mapper.dataset)):
                if class_id in classes_to_hide:
                    continue
                class_indices, labels = self.cluster_class(class_id, eps=0.4, min_points=6)
                class_name = self.mapper.class_labels[class_id]

                class_entries = class_indices.shape[0] 
                if (class_entries <= 2):
                    print(f"Skipping {class_name} viz")
                    continue

                for cluster in np.unique(labels):
                    cluster_indices = np.where(labels == cluster)[0]
                    cluster_ptc  = self.mapper.raw_point_cloud.select_by_index(class_indices[cluster_indices], invert=False)
                    aa_bb = cluster_ptc.get_axis_aligned_bounding_box()
                    # aa_bb = self.part_aligned_bounding_box(cluster_ptc)
                    aa_bb.color = self.mapper.dataset[class_name]/255
                    
                    box_min = aa_bb.get_min_bound()
                    box_max = aa_bb.get_max_bound()
                    # mark all voxels inside the bounding box with the class id
                    voxel_min = ((box_min - min_bound) / voxel_size).astype(int)
                    voxel_max = ((box_max - min_bound) / voxel_size).astype(int)
                    grid_bb = voxel_grid[voxel_min[0]:voxel_max[0], voxel_min[1]:voxel_max[1], voxel_min[2]:voxel_max[2]]

                    if class_id == 9: # windows
                        # Change walls inside the bb to windows
                        grid_bb[grid_bb == 1] = class_id
                    elif class_id == 23: # painting
                        # Flatten the paintings to the walls 
                        grid_bb[grid_bb == 1] = class_id
                    elif class_id == 15: # doors
                        # cut doors from the walls
                        grid_bb[:,:,:] = class_id 
                    elif class_id == 13: # people
                        # cut people from the walls :3
                        grid_bb[grid_bb == 1] = 0
                    else:
                        pass
                        grid_bb[:,:,:] = class_id


        # Draw the voxel grid as trimesh
        for x in range(voxel_grid.shape[0]):
            for y in range(voxel_grid.shape[1]):
                for z in range(voxel_grid.shape[2]):
                    if voxel_grid[x, y, z] != 0:
                        voxel = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
                        voxel.translate(np.array([x, y, z]) * voxel_size + min_bound)
                        voxel.paint_uniform_color(self.mapper.class_colors[voxel_grid[x, y, z]]/255)
                        vox_mesh += voxel

        self.mapper.viewer.add_geometry("VoxelGrid", vox_mesh)

        if self.mapper.trajectories:
            self.mapper.viewer.add_geometry("Trajectories", self.trajectories)

        # Create object only ptc
        hide_indices = np.where(np.isin(self.mapper.point_classes, classes_to_hide))[0]
        object_only_pcd = self.mapper.raw_point_cloud.select_by_index(hide_indices, invert=True)

        # self.geometries["Map"] = o3d.geometry.VoxelGrid.create_from_point_cloud(object_only_pcd, voxel_size)
        print("Voxel grid drawn")

        return voxel_grid

    # Random Markov Field Cut segmentation - not updated 
    def segment_interior(self):
        cutter = RMFCut()
        wall_indices = np.where(self.mapper.point_classes == 1)[0]
        wall_points = self.mapper.raw_point_cloud.select_by_index(wall_indices, invert=False)
        
        # Toy example (a box with a hole in the middle)
        ptc_voxel = np.zeros((15, 20, 15), dtype=bool)
        ptc_voxel[2:12, 2:18, 2:12]  = np.random.choice([True, False], size=(10, 16, 10), p=[0.9, 0.1])
        ptc_voxel[3:11, 3:13, 3:11]  = False
        ptc_voxel[3:11, 14:17, 3:11] = False
        ptc_voxel[6:8, :, 3:11]     = False 

        vox_points = self.mapper.raw_point_cloud.voxel_down_sample(voxel_size=0.3)
        ptc_voxel = self.fit_hor_planes(0.3)

        # Remove noise from the voxel grid
        # ptc_voxel = cutter.remove_interior_noise(ptc_voxel)
        
        cut = cutter.segment_3d_grid(ptc_voxel)
        
        pf = cutter.build_pf(cut, ptc_voxel)
        
        # Take z-axis max
        max_pf = np.max(pf[:,:,:10], axis=2)

        # Vizualize a slice of the pf
        plt.imshow(max_pf.T)
        plt.show()

    # Draw an esdf slice 
    def show_esdf(self, thres_min=0.3, thres_max=3):
        t_points = self.mapper.raw_esdf.point
        positions = t_points.positions.cpu().numpy()
        intensities = t_points.intensity.cpu().numpy()

        print("Intensity range: ", np.min(intensities), np.max(intensities))

        # Filter the points
        indicies = np.where((intensities > thres_min) & (intensities < thres_max))[0]
        positions = positions[indicies]
        positions = positions[positions[:,2] < 0.6]
        positions = positions[positions[:,2] > -0.4]

        # Create a point cloud
        esdf_pcd = o3d.geometry.PointCloud()
        esdf_pcd.points = o3d.utility.Vector3dVector(positions)
        esdf_pcd.rotate(self.mapper.esdf_rotation_matrix)
        # esdf_pcd.colors

        # Draw the point cloud
        o3d.visualization.draw_geometries([esdf_pcd])

        return esdf_pcd
    
        
    def rebuild_rooms_esdf(self):
        
        esdf_pcd = self.show_esdf(-2, 0.45)
        
        wall_voxels = self.fit_vert_planes(esdf_pcd, None, 0.3, False)

        # Preview a slice of the voxel grid
        plt.imshow(wall_voxels[:, :, 1])
        plt.show()


### VISUALIZATION ###


    def draw_objects_as_bbs(self):
        vox_mesh = o3d.geometry.TriangleMesh()

        # Iterate over all classes and draw their bounding boxes
        for class_id in range(2, len(self.mapper.dataset)):
            if class_id-1 in self.mapper.obstacle_classes or class_id-1 in self.mapper.floor_classes or class_id-1 in self.mapper.dynamic_classes:
                continue

            class_indices, labels = self.cluster_class(class_id, eps=0.4, min_points=8)
            class_name = self.mapper.class_labels[class_id]

            class_entries = class_indices.shape[0] 
            if (class_entries <= 2):
                print(f"Skipping {class_name} viz")
                continue

            for cluster in np.unique(labels):
                cluster_indices = np.where(labels == cluster)[0]
                cluster_ptc  = self.mapper.raw_point_cloud.select_by_index(class_indices[cluster_indices], invert=False)
                aa_bb = cluster_ptc.get_axis_aligned_bounding_box()
                # aa_bb = self.part_aligned_bounding_box(cluster_ptc)
                aa_bb.color = self.mapper.dataset[class_name]/255

                bb_dim = aa_bb.get_extent()
                bb_loc = aa_bb.get_center()
                voxel = o3d.geometry.TriangleMesh.create_box(width=bb_dim[0], height=bb_dim[1], depth=bb_dim[2])
                # voxel = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cluster_ptc, 1)
                voxel.translate(bb_loc-bb_dim/2)
                voxel.paint_uniform_color(self.mapper.dataset[class_name]/255)
                vox_mesh += voxel

        return vox_mesh


    def avg_nn_dist(self, n_neighbors):
        # Compute the average distance from a point to its n nearest neighbors
        distances = []
        np_points = np.asarray(self.mapper.raw_point_cloud.points)
        kd_tree = o3d.geometry.KDTreeFlann(self.mapper.raw_point_cloud)
        for i in range(len(self.mapper.raw_point_cloud.points)):
            point = self.mapper.raw_point_cloud.points[i]
            _, indices, _ = kd_tree.search_knn_vector_3d(point, n_neighbors)
            nearest_neighbors = np_points[indices[1:]]  # Exclude the point itself
            distances.append(np.mean(np.linalg.norm(nearest_neighbors - point, axis=1)))
        
        dist_mean = np.mean(distances)
        dist_std = np.std(distances)

        print(f"Average distance to {n_neighbors} nearest neighbors: {dist_mean:.2f} +/- {dist_std:.2f}")

        # plt.bar(range(len(distances)), distances, color='blue')
        # plt.xlabel('Point index')
        # plt.ylabel('Average distance to 20 nearest neighbors')
        # plt.title('Average nearest neighbor distances')
        # plt.show()

        plt.hist(distances, bins=np.logspace(np.log10(min(distances)), np.log10(max(distances)), 50), color='skyblue', edgecolor='black') #, log=True
        # plt.vlines(dist_mean, 0, 100000, colors='red', linestyles='dashed')
        plt.vlines(dist_mean+dist_std*0.00001, 0, 100000, colors='red', linestyles='dashed')
        plt.xscale('log')  # Log scale on x-axis
        plt.yscale('log')  # Optionally add log scale on y-axis
        plt.xlabel('Log-scaled mean distance to 20 nearest neighbors ')
        plt.ylabel('Log-scaled point count')
        # plt.title('Average nearest neighbor distances')
        plt.grid(True)
        plt.show()
        
        
        return np.mean(distances)

    def explode_view(self):
        if self.mapper.vis == False:
            print("Visualizer not initialized")
            return
        
        # Extract obstacle pointcloud
        obst_indices = []
        for obst_id in self.mapper.obstacle_classes:
            obst_class_indices = np.where(self.mapper.point_classes == obst_id+1)[0]
            obst_indices = np.concatenate((obst_indices, obst_class_indices), axis=0)
        obst_indices = np.unique(obst_indices).astype(int)

        # Extract floor pointcloud 
        floor_indices = []
        floor_id = 3
        floor_class_indices = np.where(self.mapper.point_classes == floor_id+1)[0]
        floor_indices = np.concatenate((floor_indices, floor_class_indices), axis=0)
        floor_indices = np.unique(floor_indices).astype(int)

        # Extract ceil pointcloud 
        ceil_indices = []
        for ceil_id in self.mapper.floor_classes:
            if ceil_id == floor_id:
                continue
            ceil_class_indices = np.where(self.mapper.point_classes == ceil_id+1)[0]
            ceil_indices = np.concatenate((ceil_indices, ceil_class_indices), axis=0)
        ceil_indices = np.unique(ceil_indices).astype(int)

        # Category statistics
        print("Obstacles: ", len(obst_indices))
        print("Floors: ", len(floor_indices))
        print("Ceilings: ", len(ceil_indices))
        print("Objects: ", len(self.mapper.raw_point_cloud.points) - len(obst_indices) - len(floor_indices) - len(ceil_indices))

        # Objects 
        non_objects = np.concatenate((obst_indices, floor_indices, ceil_indices))
        obj_classes = [14, 8]
        obj_indices = []
        for obj_id in obj_classes:
            obj_class_indices = np.where(self.mapper.point_classes == obj_id+1)[0]
            obj_indices = np.concatenate((obj_indices, obj_class_indices), axis=0)
        obj_indices = np.unique(obj_indices).astype(int)
        
        obst_ptc   = self.mapper.raw_point_cloud.select_by_index(obst_indices, invert=False)
        floor_ptc  = self.mapper.raw_point_cloud.select_by_index(floor_indices, invert=False)
        ceil_ptc   = self.mapper.raw_point_cloud.select_by_index(ceil_indices, invert=False)
        # object_ptc = self.mapper.raw_point_cloud.select_by_index(non_objects, invert=True)
        object_ptc = self.mapper.raw_point_cloud.select_by_index(obj_indices, invert=False)
        object_mesh = self.draw_objects_as_bbs()
        #  + self.mapper.floor_classes
        # wall_classes = np.array(self.mapper.obstacle_classes) + 1
        # obst_ptc = self.merge_wall_classes(wall_classes, obj_outlier_removal=False)
        obst_ptc.paint_uniform_color([0.55, 0.55, 0.55])
        # floor_ptc.paint_uniform_color([0.8, 0.8, 0.8])
        # ceil_ptc.paint_uniform_color([0.8, 0.8, 0.8])
        # object_ptc.paint_uniform_color([0.8, 0.8, 0.8])

        # self.mapper.viewer.add_geometry("Obstacles", obst_ptc)
        # self.mapper.viewer.add_geometry("Ceiling", ceil_ptc.translate([0,0,9]))
        # self.mapper.viewer.add_geometry("Floors", floor_ptc.translate([0,0,-4]))
        self.mapper.viewer.add_geometry("Objects", object_ptc.translate([0,0,6]))

