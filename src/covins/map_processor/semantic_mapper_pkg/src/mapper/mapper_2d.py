#!/usr/bin/python3

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
# import mapper.utils as utils
import struct
import time 
import yaml
import cv2



class Mapper2D(object):

    def __init__(self, mapper):
        self.geometry_names = []
        self.geometries = {}
        self.mapper = mapper

    def set_resolution(self, cell_size = 0.2, length = 50, bg_val=-2):
        self.step = cell_size
        self.img_size = length
        self.bg_val = bg_val
    

    def draw_walls(self):
        wall_layer = self.grid_counts[:,:,1] 
        non_walls = self.grid_counts[:,:,1:]
        final_grid_labels = np.argmax(non_walls, axis=2)

        thresh = 1
        wall_layer[wall_layer > thresh] = 1 # self.class_colors[wall_layer.T]
        wall_layer[wall_layer < thresh] = 0
        wall_layer[final_grid_labels > 0] = 0

        canvas_layer = wall_layer

        return canvas_layer

    def draw_objects(self, tf, id=None, classes_to_exclude = [4, 6, 9], eps=0.3, min_samples=8):
        min_bound = self.mapper.shared_min_bound
        max_bound = self.mapper.shared_max_bound
        map_size = ((max_bound - min_bound) / self.step)
        map_size = np.ceil(map_size).astype(int)[:2]
        
        obj_map = np.zeros(map_size, dtype=np.uint8)

        classes_to_include = id if id else range(2, len(self.dataset))

        for class_idx in classes_to_include:
            class_name = self.mapper.class_labels[class_idx]
            class_indices, labels = self.mapper.m3d.cluster_class(class_idx, 0.3, 8)

            class_entries = class_indices.shape[0] 
            if (class_entries <= 2 or class_idx in classes_to_exclude):
                print("Skipping ", class_name)
                continue

            for cluster in np.unique(labels):
                cluster_indices = np.where(labels == cluster)[0]
                cluster_ptc  = self.mapper.raw_point_cloud.select_by_index(class_indices[cluster_indices], invert=False)
                aa_bb = cluster_ptc.get_axis_aligned_bounding_box()
                # aa_bb = cluster_ptc.get_oriented_bounding_box()
                # aa_bb.color = self.mapper.dataset[class_name]/255
                # aa_bb = self.mapper.m3d.z_aligned_bounding_box(cluster_ptc)
                
                # Check if the bounding box is valid
                if (class_idx == 9 or class_idx == 15) and (aa_bb.volume() < 0.01 or aa_bb.volume() > 10):
                    continue
                    
                
                # aa_bb.paint_uniform_color(self.dataset[class_name]/255)
                
                # Draw the bounding box on the 2D map
                # Get the points belonging to the bounding box
                bb_points = np.asarray(aa_bb.get_box_points()) #np.asarray(aa_bb.points) #
                # Get the indices of the points belonging to the bounding box
                bb_points = (bb_points - min_bound) / self.step + tf
                bb_indices = np.floor(bb_points).astype(int) #
                # Draw the bounding box on the 2D map
                obj_map[min(bb_indices[:, 0]):max(bb_indices[:, 0]), min(bb_indices[:, 1]):max(bb_indices[:, 1])] = class_idx

                # Connect the corners of the bounding box
                for i in range(8):
                    obj_map = cv2.line(obj_map, (bb_indices[i, 1], bb_indices[i,0]), (bb_indices[(i + 1) % 8, 1], bb_indices[(i + 1) % 8, 0]), class_idx, 1)
        
        return obj_map

    def process_esdf_walls(self, walls):
        binary_walls = np.zeros_like(walls)
        binary_walls[walls != -2] = 1

        binary_walls = cv2.morphologyEx(binary_walls, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

        binary_walls = cv2.morphologyEx(binary_walls, cv2.MORPH_ERODE, np.ones((3, 3), np.uint8), iterations=1)

        return binary_walls
        


    def draw_floormap(self):        
        grid_counts = self.grid_counts

        wall_layer = grid_counts[:,:,1]
        floor_layer = grid_counts[:,:,4]
        ceil_layer = grid_counts[:,:,6]

        # # Detect walls witl sliding kernel 
        kernel = np.zeros((5,5))
        kernel[2, :] = 1 
        kernel[:, 2] = 1

        # kernel = np.ones((5,5))
        gauss_kernel = cv2.getGaussianKernel(9, 0)
        gauss_kernel = gauss_kernel @ gauss_kernel.T

        whitespace = floor_layer + ceil_layer
        # whitespace = cv2.filter2D(whitespace,-1,gauss_kernel)
        whitespace = cv2.filter2D(whitespace,-1,10*gauss_kernel)
        whitespace = cv2.threshold(whitespace, 2, 100, cv2.THRESH_BINARY)[1]
        whitespace = cv2.morphologyEx(whitespace, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        whitespace = cv2.morphologyEx(whitespace, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        whitespace = cv2.morphologyEx(whitespace, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
        whitespace = cv2.morphologyEx(whitespace, cv2.MORPH_ERODE, np.ones((3, 3), np.uint8), iterations=2)

        # Convolution
        wall_layer = cv2.filter2D(wall_layer,-1,10*gauss_kernel)
        wall_layer = cv2.threshold(wall_layer, 1, 100, cv2.THRESH_BINARY)[1]
        # Image closing
        wall_layer = cv2.morphologyEx(wall_layer, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        wall_layer = cv2.morphologyEx(wall_layer, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        wall_layer = cv2.morphologyEx(wall_layer, cv2.MORPH_ERODE, np.ones((5, 5), np.uint8), iterations=2)

        # # Gaussian blur
        # wall_layer = cv2.filter2D(wall_layer,-1,gauss_kernel)
        # # Image closing
        # # Threshold
        # wall_layer[wall_layer < 50] = 0
        wall_layer[whitespace > 1] = 5
        wall_layer = cv2.threshold(wall_layer, 10, 100, cv2.THRESH_BINARY)[1]

        # Take the argmax over the 3rd dimension to get a 2D map of class ids
        final_grid_labels = np.argmax(grid_counts, axis=2)
        
        canvas_layer = wall_layer

        # Plot the 2D projection as an image
        fig, ax = plt.subplots()
        im = ax.imshow(canvas_layer.T, origin='lower')
        # im = ax.imshow(self.class_colors[final_grid_labels.T], origin='lower')
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.set_title('2D Projection with Most Common Class')
        fig.colorbar(im, ax=ax, label='Class')
        # leg = ax.legend(fancybox=True)

        plt.show()
        exit()

    def compute_covisibility(self, binary_image):
        """
        Computes the covisibility of positive pixels in a binary image.

        Args:
            binary_image (numpy.ndarray): A 2D binary image with 0s and 1s.

        Returns:
            numpy.ndarray: A 2D matrix where each row represents a positive pixel,
                        and each column indicates the covisibility with other pixels.
        """
        rows, cols = binary_image.shape
        positive_pixels = np.argwhere(binary_image == 1)
        num_positive_pixels = len(positive_pixels)

        covisibility_matrix = np.zeros((num_positive_pixels, num_positive_pixels), dtype=int)

        for i in range(num_positive_pixels):
            for j in range(i + 1, num_positive_pixels):
                pixel1 = positive_pixels[i]
                pixel2 = positive_pixels[j]

                # Check if the two pixels are covisible using Bresenham's line algorithm
                if self.is_covisible(binary_image, pixel1, pixel2):
                    covisibility_matrix[i, j] = 1
                    covisibility_matrix[j, i] = 1

        return covisibility_matrix


    def is_covisible(self, binary_image, pixel1, pixel2):
        """
        Checks if two pixels are covisible using Bresenham's line algorithm.

        Args:
            binary_image (numpy.ndarray): A 2D binary image with 0s and 1s.
            pixel1 (tuple): Coordinates of the first pixel (row, col).
            pixel2 (tuple): Coordinates of the second pixel (row, col).

        Returns:
            bool: True if the pixels are covisible, False otherwise.
        """
        x1, y1 = pixel1
        x2, y2 = pixel2

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1

        err = dx - dy

        while x1 != x2 or y1 != y2:
            if binary_image[x1, y1] == 0:
                return False

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return True

                
    def cluster_watershed(self, grid):
        from skimage.filters import threshold_otsu
        # Binarise
        thresh = threshold_otsu(grid)
        binary = grid.copy()
        binary[binary>thresh] = 1
        binary[binary<=thresh] = 0 


        coords = peak_local_max(grid, min_distance=20, threshold_abs=1.1, footprint=np.ones((3, 3)), labels=binary)
        mask = np.zeros(grid.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-grid, markers, mask=binary)



        return labels


    def construct_2d_esdf(self, compression="max_abs"):
        voxel_size = self.step
        if type(self.mapper.raw_esdf) == o3d.core.Tensor:
            # Tensor  to numpy
            t_points = self.mapper.raw_esdf.point
            # R = self.mapper.esdf_rotation_matrix
            # positions = ( R @ t_points.positions.cpu().numpy().T ).T
            positions = t_points.positions.cpu().numpy()
            intensities = t_points.intensity.cpu().numpy()
        else:
            positions = np.asarray(self.mapper.raw_esdf.points)
            intensities =  np.asarray(self.mapper.raw_esdf.colors)[:,0]
        
        # min_bound = np.min(positions, axis=0)
        # max_bound = np.max(positions, axis=0)
        min_bound = self.mapper.shared_min_bound
        max_bound = self.mapper.shared_max_bound
        map_size = ((max_bound - min_bound) / voxel_size)
        map_size = np.ceil(map_size).astype(int)[:2]
        
        bg_i = self.bg_val
        esdf_grid   = np.ones(map_size) * bg_i
        count_grid  = np.zeros(map_size)
        height_grid = np.zeros(map_size)
        
        max_z = np.max(positions[:, 2])
        min_z = np.min(positions[:, 2])
        mid_z = (max_z + min_z)/2
        z_window = 3 # meters
        print(max_z, min_z, mid_z)
        
        for i in range(positions.shape[0]):
            x, y, z = positions[i]
            intensity = intensities[i]
            x_idx = int((x - min_bound[0])/voxel_size)
            y_idx = int((y - min_bound[1])/voxel_size)
            z_idx = int((z - min_bound[2])/voxel_size)
            
            if not (mid_z - z_window < z < mid_z + z_window):
                continue

            if count_grid[x_idx, y_idx] == 0:
                esdf_grid[x_idx, y_idx] = 0
            count_grid[x_idx, y_idx] += 1
            
            if compression=="sum" or compression=="mean":
                esdf_grid[x_idx, y_idx] += intensity
                height_grid[x_idx, y_idx] = z_idx       # probably worsens the result actually 
            if compression=="max_abs" and abs(intensity) > abs(esdf_grid[x_idx, y_idx]):
                esdf_grid[x_idx, y_idx] = intensity
                height_grid[x_idx, y_idx] = z_idx
            if compression=="max" and intensity > esdf_grid[x_idx, y_idx]:
                esdf_grid[x_idx, y_idx] = intensity
                height_grid[x_idx, y_idx] = z_idx
            if compression=="min" and intensity < esdf_grid[x_idx, y_idx]:
                esdf_grid[x_idx, y_idx] = intensity
                height_grid[x_idx, y_idx] = z_idx
                           
            
        # Average the grid by the number of points in each cell
        if compression=="mean":
            esdf_grid   = np.divide(esdf_grid, count_grid, out=np.ones_like(esdf_grid)*bg_i, where=count_grid!=0)
        
        # Fill in the empty cells with the average of the surrounding cells
        for i in range(1, map_size[0]-1):
            for j in range(1, map_size[1]-1):
                if esdf_grid[i, j] != bg_i:
                    continue
                # Surrounding window
                w_size = 1
                window = esdf_grid[(i-w_size):(i+w_size+1), (j-w_size):(j+w_size+1)]
                if np.sum(window == bg_i) > 1:
                    continue
                esdf_grid[i, j] = np.mean(window[window != bg_i])
                count_grid[i, j] = 1
                height_values = height_grid[(i-w_size):(i+w_size+1), (j-w_size):(j+w_size+1)]
                height_grid[i, j] = np.mean(height_values[height_values!=0])


        return esdf_grid, count_grid, height_grid


    def cluster_dbscan(self, show=False, w_spat=0.6, w_int=0.4, esdf_grid=None, count_grid=None, height_grid=None):
        '''
        Use raw_esdf to generate 2d map of rooms 
        '''
        gt_esdf = False
        if esdf_grid is None:
            esdf_grid, count_grid, height_grid = self.construct_2d_esdf(compression="max_abs")
        if gt_esdf:
            esdf_grid = self.mapper.m3d.test_esdf_voxel_grid(0.2, True)
            count_grid = np.zeros_like(esdf_grid)
            count_grid[esdf_grid != self.bg_val] = 1
            height_grid = np.zeros_like(esdf_grid) # ignoring for now 
        bg_i = self.bg_val
        # grid = (grid - np.min(grid))/(np.max(grid) - np.min(grid))
    
        grid_var = np.max(esdf_grid) - np.min(esdf_grid)
        grid_steps = grid_var/self.step * 2
        print("ESDF variance: ", grid_var, "Distance steps: ", grid_steps)
        
        # Build intensity histogram
        hist, bins = np.histogram(esdf_grid[count_grid!=0].flatten(), bins=int(grid_steps))
        # Most common intensity
        thres_intensity = bins[np.argmax(hist)] 
        print("Threshold intensity: ", thres_intensity)
        
        grid = esdf_grid.copy()
        grid[esdf_grid < thres_intensity] = bg_i
        if show:
            self.draw_intensity_binarization(esdf_grid, thres_intensity, bg_i)
                
        walls_grid = esdf_grid.copy()
        walls_grid[esdf_grid > thres_intensity] = bg_i
        # walls_grid[esdf_grid <= thres_intensity] = 5
        # walls_grid = self.process_esdf_walls(walls_grid)

        # Flatten the image into a 2D array of pixel coordinates and intensities
        pixels_flat = np.column_stack((np.where(grid >= thres_intensity)))
        intens_flat = grid[pixels_flat[:, 0], pixels_flat[:, 1]] 
        z_flat = height_grid[pixels_flat[:, 0], pixels_flat[:, 1]]
        
        # Pixel distance matrix function 
        def intensity_distance(coords):
            spatial_distance = pdist(coords[:, :3], metric='euclidean')
            intensity_distance = pdist(coords[:, 3].reshape(-1, 1), metric='euclidean')
            combined_distance = w_spat*spatial_distance + w_int*intensity_distance
            return squareform(combined_distance)

        # Create a feature matrix by combining pixel coordinates and normalized intensities
        features = np.column_stack((pixels_flat, z_flat, intens_flat))
        # Compute the distance matrix using the custom distance metric
        distance_matrix = intensity_distance(features)
        print("Constructed distance matrix ")
        
        # Perform HDBSCAN clustering
        hdbscan = HDBSCAN(min_cluster_size=100, min_samples=4, metric='precomputed')
        labels = hdbscan.fit_predict(distance_matrix)
        print("Clustering complete ")
            
        # Get the unique cluster labels (excluding noise points)
        unique_labels = np.unique(labels)
        
        # Create a segmented image based on the cluster labels
        segmented_image = np.ones_like(grid)*bg_i
        for label in unique_labels:
            mask = (labels == label)
            segmented_image[pixels_flat[mask, 0], pixels_flat[mask, 1]] = label 

            
        # Relable the noise points
        for i in range(1):
            for noise_pix in pixels_flat[labels==-1]:
                # Surrounding window
                w_size = 1
                vote_window = segmented_image[(noise_pix[0]-w_size):(noise_pix[0]+w_size+1), 
                                                (noise_pix[1]-w_size):(noise_pix[1]+w_size+1)]
                windows_classes = [np.sum(vote_window==c) if c!=-1 else 0 for c in unique_labels]
                max_class = np.argmax(windows_classes)
                # relabel 
                if windows_classes[max_class] >= 4:
                    segmented_image[noise_pix[0], noise_pix[1]] = max_class-1
        # walls_grid[labels==-1] = 1      
                
        # Labeling the rest  
        unlabled_grid = esdf_grid.copy()
        unlabled_grid[unlabled_grid>thres_intensity] = bg_i
        grid_var = np.max(unlabled_grid) - np.min(unlabled_grid)
        grid_steps2 = grid_var/self.step
        hist, bins = np.histogram(unlabled_grid[unlabled_grid!=bg_i].flatten(), bins=int(grid_steps2))
        # Most common intensity 2
        sec_thres_intensity = bins[np.argmax(hist)] 
        # walls_grid[esdf_grid > sec_thres_intensity] = bg_i
        print("Second threshold intensity: ", sec_thres_intensity)  # thres_intensity=-0.28
        if show:
            plt.hist(esdf_grid[count_grid!=0].flatten(), bins=int(grid_steps))
            plt.axvline(thres_intensity, color='r')
            plt.axvline(sec_thres_intensity, color='r')
            plt.show()
        
        other_pixels_flat = np.column_stack((np.where(unlabled_grid >= sec_thres_intensity)))
        # Relable the obst points
        for i in range(4):
            relabeled = 0
            for noise_pix in other_pixels_flat:
                # Surrounding window
                w_size = 1
                vote_window = segmented_image[(noise_pix[0]-w_size):(noise_pix[0]+w_size+1), 
                                                (noise_pix[1]-w_size):(noise_pix[1]+w_size+1)]
                windows_classes = [np.sum(vote_window==c) if c!=-1 else 0 for c in unique_labels]
                max_class = np.argmax(windows_classes)
                # relabel 
                if windows_classes[max_class] >= 4:
                    relabeled += 1
                    segmented_image[noise_pix[0], noise_pix[1]] = max_class-1

            if relabeled == 0:
                break

        # Remove unlabeled stuff
        segmented_image[segmented_image == -1] = bg_i
        
        # Viz the layers 
        esdf_sep = esdf_grid.copy()
        # HACK: FOR AN IMAGE - REMOVE
        # segmented_image = esdf_grid >= sec_thres_intensity

        # first_sep = esdf_grid >= thres_intensity
        # first_neg_sep = (esdf_grid > bg_i) & (esdf_grid < sec_thres_intensity)
        # # sec_sep = esdf_grid >= sec_thres_intensity
        # # esdf_sep[sec_sep] = 10
        # first_neg_sep = cv2.morphologyEx(first_neg_sep.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1).astype(np.bool)
        # # first_neg_sep = cv2.ximgproc.thinning(first_neg_sep, first_neg_sep, cv2.ximgproc.THINNING_GUOHALL)
        # from skimage.morphology import skeletonize, thin
        # first_neg_sep = skeletonize(first_neg_sep)
        # # first_neg_sep = thin(first_neg_sep)
        # esdf_sep[first_neg_sep>0] = 5
        # esdf_sep[first_sep] = 0
        # walls_grid = first_neg_sep>0

        if show:
            # Plot the original and segmented images
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            im = ax[0].imshow(esdf_sep, cmap='viridis')
            plt.colorbar(im, ax=ax[0])
            ax[0].set_title('Intensity Map')
            ax[1].imshow(segmented_image, cmap='tab20b')
            ax[1].set_title('Segmented Image')
            plt.show()
        
            
        return segmented_image, walls_grid, esdf_grid
    

    def fuse_pointclouds(self, tf, show=False):
        # self.build_grid(voxel_size, length)
        
        # wall_layer = self.draw_walls().astype(np.uint8)
        room_layer, wall_layer, esdf_layer = self.cluster_dbscan(False, 0.6, 0.2)
        # wall_layer = wall_layer > self.bg_val
        # Erase noise 
        # room_layer[room_layer==-1] = -2
        
        obj_layer = self.draw_objects(tf, [9, 15], [4, 6])

        # Colormaps
        from matplotlib.colors import ListedColormap, Normalize
        # Define colors for free/occupied space (white for free, black for occupied)
        free_occupied_cmap = ListedColormap(['white','black'])

        # Define colors for room clusters using 'tab20' colormap
        tab20 = plt.cm.get_cmap('tab10', 10)  # Get 10 colors from tab20
        transparent = [0, 0, 0, 0]  # RGBA for fully transparent
        room_clusters_colors = [tab20(i) for i in range(10)]
        room_clusters_colors[0] = transparent
        room_clusters_cmap = ListedColormap(room_clusters_colors)

        room_layer[room_layer>=0]+=1
        # Map room_clusters values, adjusting for transparent indexing
        room_clusters_mapped = np.where(room_layer == -2, 0, room_layer )

        # Define specific colors for objects, including RGBA for consistency
        objects_colors = np.array([
            [1, 1, 1, 0],      # Transparent for value 0 (so room and free/occupied can show through)
            [1, 0.753, 0.796, 1],  # Light pink with full opacity
            [0, 1, 0, 1]       # Green with full opacity
        ])
        objects_cmap = ListedColormap(objects_colors)
        objects_norm = Normalize(vmin=0, vmax=2)

        # Map objects values to indices 0, 1, 2
        objects_mapped = np.zeros_like(obj_layer)
        objects_mapped[obj_layer == 9] = 1
        objects_mapped[obj_layer == 15] = 2

        # Combine the layers, starting with free/occupied
        combined = np.where(wall_layer == -2, 0, 12)  # Free to 11, occupied to 12
        # Overlay room clusters
        combined = np.where(room_clusters_mapped > 0, room_clusters_mapped + 1, combined)
        # Overlay objects
        combined = np.where(objects_mapped > 0, objects_mapped + 12, combined)  # Ensure unique mapping

        # Combined colormap
        combined_colors = [transparent] + room_clusters_colors[:] + ['white', 'black'] + list(objects_colors[1:])
        combined_cmap = ListedColormap(combined_colors)
        combined_norm = Normalize(vmin=0, vmax=len(combined_colors) - 1)

        # Visualization
        if show:
            fig, ax = plt.subplots(1, 4, figsize=(25, 5))
            ax[0].imshow(wall_layer, cmap=free_occupied_cmap, interpolation='nearest')
            ax[0].set_title('Free/Occupied Space')
            ax[1].imshow(room_clusters_mapped, cmap=room_clusters_cmap, interpolation='nearest')
            ax[1].set_title('Room Clusters')
            ax[2].imshow(objects_mapped, cmap=objects_cmap, norm=objects_norm, interpolation='nearest')
            ax[2].set_title('Objects')
            ax[3].imshow(combined, cmap=combined_cmap, norm=combined_norm, interpolation='nearest')
            ax[3].set_title('Combined Map')

            plt.tight_layout()
            plt.show()

        # Use the colormap and the normalizer to convert the data to an RGB image
        img = combined_cmap(combined_norm(combined))

        # The image will be in the range [0, 1], so we scale it to [0, 255]
        data = (img * 255).astype(np.uint8)

        return data

    def draw_esdf_slice(self, height=None):
        voxel_size = self.step
        # Tensor  to numpy
        t_points = self.mapper.raw_esdf.point
        # R = self.mapper.esdf_rotation_matrix
        # positions = ( R @ t_points.positions.cpu().numpy().T ).T
        positions = t_points.positions.cpu().numpy()
        intensities = t_points.intensity.cpu().numpy()
        
        # min_bound = np.min(positions, axis=0)
        # max_bound = np.max(positions, axis=0)
        min_bound = self.mapper.shared_min_bound
        max_bound = self.mapper.shared_max_bound
        map_size = ((max_bound - min_bound) / voxel_size)
        map_size = np.ceil(map_size).astype(int)[:2]
        
        bg_i = self.bg_val
        esdf_grid   = np.ones(map_size) * bg_i
        count_grid  = np.zeros(map_size)
        height_grid = np.zeros(map_size)
        
        max_z = np.max(positions[:, 2])
        min_z = np.min(positions[:, 2])
        mid_z = height if height else int(((max_z + min_z)/2 - min_bound[2])/voxel_size)
        print(mid_z)
        
        for i in range(positions.shape[0]):
            x, y, z = positions[i]
            intensity = intensities[i]
            x_idx = int((x - min_bound[0])/voxel_size)
            y_idx = int((y - min_bound[1])/voxel_size)
            z_idx = int((z - min_bound[2])/voxel_size)  

            if z_idx != mid_z+2:
                continue

            esdf_grid[x_idx, y_idx] = intensity

        
        # plt.figure()
        # plt.imshow(esdf_grid)
        # plt.axis('equal')
        # plt.colorbar()
        # plt.show()



    def draw_intensity_binarization(self, esdf_grid, thres_intensity, bg_i):
        # Figure with esdf_map, histogram and grid after thresholding
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        im = ax[0].imshow(esdf_grid, cmap='viridis')
        plt.colorbar(im, ax=ax[0])
        ax[0].set_title('Intensity Map')
        
        ax[1].hist(esdf_grid[esdf_grid!=bg_i].flatten(), bins=23)
        ax[1].set_title('Intensity Histogram')
        ax[1].axvline(thres_intensity, color='r', linestyle='dashed', linewidth=2)
        
        disp_grid = esdf_grid.copy()
        disp_grid[esdf_grid <= thres_intensity] = 2
        disp_grid[esdf_grid > thres_intensity] = 0
        disp_grid[esdf_grid == bg_i] = bg_i
        im = ax[2].imshow(disp_grid, cmap='viridis')
        plt.colorbar(im, ax=ax[2])
        ax[2].set_title('Thresholded Intensity Map')
        
        plt.tight_layout()
        plt.show()     