import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image
import matplotlib.image

from mapper import Mapper

# dataset_path = r"C:\Users\Matvey\Repos\covins_demo\src\covins\covins_backend\config\objects.yaml" 
# pcd_path =  r"C:\Users\Matvey\Repos\covins_demo\output\covins_pcds\clc12_ag112_tuned.pcd" 
# esdf_path = r"C:\Users\Matvey\Repos\covins_demo\output\voxblox_pcds\vox_esdf_02m_clc12_ag112_tuned.pcd"
dataset_path = "/home/matvei/repos/covins_demo/src/covins/covins_backend/config/objects.yaml" 
pcd_path = "/home/matvei/repos/covins_demo/output/covins_pcds/clc12_ag112_tuned.pcd" 
esdf_path = "/home/matvei/repos/covins_demo/output/voxblox_pcds/vox_esdf_02m_clc12_ag112_tuned_new_rotated.pcd"

# pcd_path = "/home/matvei/repos/covins_demo/output/covins_pcds/full_two_agents_gba.pcd"
# esdf_path = "/home/matvei/repos/covins_demo/output/voxblox_pcds/vox_esdf_02m_long_update_obsts_ext_filtered.pcd"

pcd = o3d.io.read_point_cloud(pcd_path)
esdf = o3d.t.io.read_point_cloud(esdf_path)
tf = [0, 5, 0]

mpr = Mapper(vis=False)
mpr.load_dataset(dataset_path) # 35 15
mpr.setup_point_clouds(pcd, "Map", 14, False)
mpr.setup_intensity_map(esdf, "ESDF", 1, False)
# viewer3d.load_trajectories(["traj1.pcd",
#                             "traj2.pcd"], -20)

mpr.m2d.set_resolution(0.2, 60, -2)
# mpr.m3d.explode_view()
# mpr.m3d.avg_nn_dist(50)
# segmented_img, _, esdf_grid = mpr.m2d.cluster_dbscan(True, 0.5, 0.8)
# matplotlib.image.imsave('seg_rooms_max.png', segmented_img)

# mpr.m2d.fuse_pointclouds(tf)
# mpr.m3d.show_esdf(-8, 8)
# mpr.m3d.show_esdf(-2, 0.496)
# mpr.m2d.draw_esdf_slice()
# mpr.m3d.rebuild_rooms_esdf()
# mpr.m3d.rebuild_rooms() 
# mpr.m3d.draw_class_distribution()

# mpr.draw_nn_perf()
# mpr.val_esdf()
# mpr.test_cluster_params()
# mpr.compare_esdf_compression()

# viewer3d.create_voxel_grid()
# viewer3d.rebuild_rooms()
# viewer3d.cluster_all_classes()
# viewer3d.segment_interior()
# viewer3d.reconstruct_surface()
# viewer3d.draw_class_distribution()

# with Image.open("/home/matvei/repos/covins_demo/mapper/seg_rooms_gt.png") as img:
#         grayscale = img.convert('L')  # Convert to grayscale
#         gt_map =  np.array(grayscale)
# gt_map = (gt_map < 255).astype(np.uint8)
# # gt_labels = mpr.get_unique_clusters(gt_map)
# scores, shifts, max_score = mpr.find_max_iou(gt_map, 10)
# print("Max full map IoU: ", max_score)

# # seg_labels = np.unique(segmented_img)
# # iou_scores = mpr.calculate_iou_per_cluster(gt_map, segmented_img, gt_labels, seg_labels)

try:
    while mpr.vis:        
        # mpr.m3d.filter_point_clouds(voxel=True)
        # viewer3d.label_points()
        # Step 2) Update the cloud and tick the GUI application
        mpr.viewer.update_o3d_scene()
        mpr.viewer.run_one_tick()
except Exception as e:
    print(e)
