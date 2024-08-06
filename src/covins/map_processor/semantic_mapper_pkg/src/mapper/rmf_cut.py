import maxflow
import re
import networkx as nx
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


class RMFCut:
    def __init__(self):
        pass
    
    def build_raw_occupancy_voxel_grid(self, pcd, voxel_size) -> np.ndarray:
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        voxel_grid = np.zeros((int((max_bound[0] - min_bound[0]) / voxel_size)+1,
                               int((max_bound[1] - min_bound[1]) / voxel_size)+1,
                               int((max_bound[2] - min_bound[2]) / voxel_size)+1), dtype=bool)

        for point in pcd.points:
            voxel_coord = ((point - min_bound) / voxel_size).astype(int)
            voxel_grid[voxel_coord[0], voxel_coord[1], voxel_coord[2]] = True

        return voxel_grid
    
    
    def remove_interior_noise(self, voxel_grid:np.ndarray) -> np.ndarray:
        # Find busy-free-busy-free-busy patterns in the z direction and remove the central busy voxel

        for x in range(voxel_grid.shape[0]):
            for y in range(voxel_grid.shape[1]):
                
                z_seq = voxel_grid[x, y, :]
                # Find the pattern
                sequence_str = ''.join(map(str, z_seq.astype(int)))

                # Define the pattern to match
                pattern = r'0*1+0*1+0*'

                # Use regex to find and replace the middle group
                result_str = re.sub(pattern, '0', sequence_str)
                result_list = list(result_str)

                # Convert result string back to boolean numpy array
                voxel_grid[x, y, :] = np.array(list(map(int, result_list)), dtype=bool)

        return voxel_grid

                 


    def create_graph(self, voxel_grid:np.ndarray) -> (maxflow.Graph[float], np.ndarray):
        # Create a graph with nodes corresponding to voxels
        graph = maxflow.Graph[float]()
        
        nodeids = graph.add_grid_nodes(voxel_grid.shape)
        
        # Add edges between neighboring voxels
        beta = 0.6  # Smoothness term scalar
        # graph.add_grid_edges(nodeids, beta, symmetric=True)
        # Add normal edges between unoccupied voxels
        for x in range(voxel_grid.shape[0]):
            for y in range(voxel_grid.shape[1]):
                for z in range(voxel_grid.shape[2]):
                    if voxel_grid[x, y, z]:
                        continue
                    if x > 0 and not voxel_grid[x - 1, y, z]:
                        graph.add_edge(nodeids[x, y, z], nodeids[x - 1, y, z], beta, beta)
                    if y > 0 and not voxel_grid[x, y - 1, z]:
                        graph.add_edge(nodeids[x, y, z], nodeids[x, y - 1, z], beta, beta)
                    if z > 0 and not voxel_grid[x, y, z - 1]:
                        graph.add_edge(nodeids[x, y, z], nodeids[x, y, z - 1], beta, beta)
        
        
        return graph, nodeids
    

    def compute_data_term(self, voxel_grid:np.ndarray, voxel_index:tuple, w:list) -> float:
        # Get the indices of the 6 neighboring voxels
        x, y, z = voxel_index
        
        def ev(direction):
            # Get the occupancy of the neighboring voxel
            x, y, z = voxel_index
            nx, ny, nz = x + direction[0], y + direction[1], z + direction[2]
            # Check the whole direction
            while 0 <= nx < voxel_grid.shape[0] and 0 <= ny < voxel_grid.shape[1] and 0 <= nz < voxel_grid.shape[2]:
                if voxel_grid[nx, ny, nz]:
                    return True
                nx += direction[0]
                ny += direction[1]
                nz += direction[2]
            
            return False
        
        # Collect evidence for the voxel to be exterior
        evidence = w[0] *  ev((0,0,-1)) + w[1] * ev((0,0,1)) + \
                   w[2] * (ev((0,0,-1)) and  ev((0,0,1))) + \
                   w[3] * (ev((-1,0,0)) and  ev((1,0,0))) + \
                   w[4] * (ev((0,-1,0)) and  ev((0,1,0)))
            
        return evidence
    
    def plot_graph_3d(self, graph, nodes_shape, plot_terminal=True, plot_weights=True, font_size=7):
        w_h = nodes_shape[1] * nodes_shape[2]
        X, Y = np.mgrid[:nodes_shape[1], :nodes_shape[2]]
        aux = np.array([Y.ravel(), X[::-1].ravel()]).T
        positions = {i: v for i, v in enumerate(aux)}

        for i in range(1, nodes_shape[0]):
            for j in range(w_h):
                positions[w_h * i + j] = [positions[j][0] + 0.3 * i, positions[j][1] + 0.2 * i]

        positions['s'] = np.array([-1, nodes_shape[1] / 2.0 - 0.5])
        positions['t'] = np.array([nodes_shape[2] + 0.2 * nodes_shape[0], nodes_shape[1] / 2.0 - 0.5])

        nxg = graph.get_nx_graph()
        if not plot_terminal:
            nxg.remove_nodes_from(['s', 't'])

        nx.draw(nxg, pos=positions)
        nx.draw_networkx_labels(nxg, pos=positions)
        if plot_weights:
            edge_labels = dict([((u, v), d['weight']) for u, v, d in nxg.edges(data=True)])
            nx.draw_networkx_edge_labels(nxg,
                                        pos=positions,
                                        edge_labels=edge_labels,
                                        label_pos=0.3,
                                        font_size=font_size)
        plt.axis('equal')
        plt.show()
    
    def segment_voxel_grid(self, voxel_grid):
        graph, nodeids = self.create_graph(voxel_grid)
        weights = [0.43, 0.1425, 0.1425, 0.1425, 0.1425] # Weights for the neighbors from the paper
        max_ev = sum(weights)   # practically 1
        
        # Set the data term for each voxel
        for x in range(voxel_grid.shape[0]):
            for y in range(voxel_grid.shape[1]):
                for z in range(voxel_grid.shape[2]):
                    # Skip the voxel if it is occupied
                    if voxel_grid[x, y, z]:
                        continue
                    evidence = self.compute_data_term(voxel_grid, (x, y, z), weights)
                    # graph.add_tedge(nodeids[x, y, z], evidence, 1 - evidence)
                    graph.add_tedge(nodeids[x, y, z],  1 - evidence, evidence)
                    
        # self.plot_graph_3d(graph, nodeids.shape, plot_terminal=True, plot_weights=True)
        
        # Solve the min-cut problem
        graph.maxflow()
        
        # Get the segmentation result
        segments = graph.get_grid_segments(nodeids)
        
        return segments

    def segment_3d_grid(self, ptc_voxel, show=False):
        
        segments = self.segment_voxel_grid(ptc_voxel)
        
        # Preview a cross section of the voxel grid
        section = ptc_voxel[:, :, int(ptc_voxel.shape[2] / 2)]
        seg_section = segments[:, :, int(ptc_voxel.shape[2] / 2)]

        # Compare the original and segmented voxel grid
        fig, ax = plt.subplots(1, 2)
        im = ax[0].imshow(section.T, origin='lower')
        ax[0].set_title('Original Voxel Grid')
        fig.colorbar(im, ax=ax[0], label='Occupancy')
        im = ax[1].imshow(seg_section.T, origin='lower')
        ax[1].set_title('Segmented Voxel Grid')
        fig.colorbar(im, ax=ax[1], label='Segment')
        plt.show()
        
        if show:
            voxel_array = segments | ptc_voxel
            colors = np.empty(voxel_array.shape, dtype=object)
            colors[segments] = 'blue'
            colors[ptc_voxel] = 'red'
            ax = plt.figure().add_subplot(projection='3d')
            ax.voxels(voxel_array, edgecolor='k', facecolors=colors,  alpha=0.5)
            plt.show()
        
        return segments

    # Find the positive z half-space nearest 3D neighbor distance 
    def get_nn_distance(self, voxel_grid:np.ndarray, voxel_index:tuple) -> float:
        x0, y0, z0 = voxel_index
        min_distance = np.inf
        # TODO: optimize this
        for x in range(voxel_grid.shape[0]):
            for y in range(voxel_grid.shape[1]):
                for z in range(z0+1, voxel_grid.shape[2]):
                    if not voxel_grid[x, y, z]:
                        continue
                    distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
                    min_distance = min(min_distance, distance)
                    if min_distance <= 1:
                        return min_distance
        return min_distance
        
        
    
    def build_pf(self, indoor_space:np.ndarray, voxel_grid:np.ndarray) -> np.ndarray:
        # Create a point field from the voxel grid
        pf = np.zeros(voxel_grid.shape, dtype=float)
        print("Building the Point Field ... ")
        for x in range(indoor_space.shape[0]):
            for y in range(indoor_space.shape[1]):
                for z in range(indoor_space.shape[2]-1, 0, -1):
                    if not indoor_space[x, y, z]:
                        continue # No need to compute the distance for outside voxels
                    if voxel_grid[x, y, z]:
                        continue # Occupied voxels remain 0
                    pf[x, y, z] = self.get_nn_distance(voxel_grid, (x, y, z))
                    
        return pf