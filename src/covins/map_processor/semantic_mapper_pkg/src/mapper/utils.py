#!/usr/bin/python3

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def unpack_rgb(rgb_packed):
    # Unpack RGB from packed uint32_t
    r = (rgb_packed >> 16) & 0x0000ff
    g = (rgb_packed >> 8)  & 0x0000ff
    b = (rgb_packed)       & 0x0000ff
    return r, g, b

def get_top_colors(color_array, top_n):

    # Convert 3D array to 2D (flattening the array)
    rgb_array_flat = color_array.reshape(-1, 3)

    # Get unique colors and their counts
    unique_colors, counts = np.unique(rgb_array_flat, axis=0, return_counts=True)

    # Sort based on counts
    sorted_indices = np.argsort(-counts)  # Sort in descending order
    top_5_indices = sorted_indices[:top_n]    # Get top 5 indices

    # Get top 5 colors and their counts
    top_5_colors = unique_colors[top_5_indices]
    top_5_counts = counts[top_5_indices]

    print("Top 5 most common colors:")
    for color, count in zip(top_5_colors, top_5_counts):
        print(f"Color: {color}, Count: {count}")



def get_colors_indicies(color_array, colors_to_find):
    # Initialize list to store indices for each color
    indices_list = []

    # Find indices where the specified colors occur
    for color_to_find in colors_to_find:
        indices = np.where((color_array == color_to_find).all(axis=1))[0]
        indices_list.append(indices)

    return np.concatenate(indices_list )

def draw_intensity_binarization(esdf_grid, thres_intensity, bg_i):
        # Figure with esdf_map, histogram and grid after thresholding
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        im = ax[0].imshow(esdf_grid, cmap='viridis')
        plt.colorbar(im, ax=ax[0])
        ax[0].set_title('Intensity Map')
        
        ax[1].hist(esdf_grid[esdf_grid!=bg_i].flatten(), bins=23)
        ax[1].set_title('Intensity Histogram')
        ax[1].axvline(thres_intensity, color='r', linestyle='dashed', linewidth=1)
        
        disp_grid = esdf_grid.copy()
        disp_grid[esdf_grid <= thres_intensity] = 2
        disp_grid[esdf_grid > thres_intensity] = 0
        disp_grid[esdf_grid == bg_i] = bg_i
        im = ax[2].imshow(disp_grid, cmap='viridis')
        plt.colorbar(im, ax=ax[2])
        ax[2].set_title('Thresholded Intensity Map')
        
        plt.tight_layout()
        plt.show()     