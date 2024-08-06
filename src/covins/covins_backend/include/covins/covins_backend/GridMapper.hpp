#pragma once

#include <string>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>

#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_octomap/GridMapOctomapConverter.hpp>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// ROS
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <dynamic_reconfigure/server.h>
#include <covins_backend/GridMapperConfig.h>

// COVINS
#include "covins_backend/visualization_be.hpp"
#include "covins_backend/keyframe_be.hpp"
#include "covins_backend/landmark_be.hpp"

//Octomap
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/GetOctomap.h>
#include <octomap_msgs/BoundingBoxQuery.h>
#include <octomap_msgs/conversions.h>

#include <octomap_ros/conversions.h>
#include <octomap/octomap.h>
#include <octomap/OcTreeKey.h>
#include <octomap/ColorOcTree.h>

// struct PointXYZL {
//     PCL_ADD_POINT4D;
//     int label;
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// } EIGEN_ALIGN16;

// POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZL,
//                                   (float, x, x)
//                                   (float, y, y)
//                                   (float, z, z)
//                                   (int, label, label))


using namespace cv;
using namespace grid_map;
using namespace octomap;

class GridMapper {
public:
    GridMapper(const ros::NodeHandle private_nh_ = ros::NodeHandle("~"));
    ~GridMapper() = default;

    typedef octomap::ColorOcTree OcTreeT; // or color 

    void reconfigureCallback(covins_backend::GridMapperConfig& config, uint32_t level);

    void insertKF(covins::VisualizerBase::KeyframePtr kf);

    pcl::PointCloud<pcl::PointXYZL>::Ptr createLabeledCloud( covins::VisualizerBase::LandmarkMap &landmarks );
    pcl::PointIndices::Ptr removeFurnitureNoise( const pcl::PointCloud<pcl::PointXYZL>::Ptr &cloud, std::list<int> &lm_ids,
        const std::vector<int> &obstacleLabels, const std::vector<int> &ignoreLabels, float clusterTolerance, int minClusterSize, int maxClusterSize);
    void resetMap();
    void printTreeParams();
    void publishMapAsMarkers(const ros::Time& rostime, bool publishMarkerArray); 
    void publishFullOctoMap(const ros::Time& rostime) const;
    void publishBinaryOctoMap(const ros::Time& rostime) const;
    void publishProjected2DMap(const ros::Time& rostime) ;
    bool populateGridMap();
    void axisDirCallback(const geometry_msgs::PoseWithCovarianceStamped& wall_axis);
    void updateTransform();


    grid_map::GridMap map;

    // octomap
    double oc_resolution;
    int m_treeDepth;
    int m_maxTreeDepth;
    double m_minRange;
    double m_maxRange;
    bool m_useColoredMap;
    bool m_publish2DMap; 
    std::string m_worldFrameId;

    OcTreeT* m_octree;
    KeyRay m_keyRay;  // temp storage for ray casting
    OcTreeKey m_updateBBXMin;
    OcTreeKey m_updateBBXMax;

    nav_msgs::OccupancyGrid m_gridmap;
    octomap::OcTreeKey m_paddedMinKey;
    bool m_projectCompleteMap;
    unsigned m_multires2DScale;

    Eigen::Matrix3d axisRotMatrix;

    // ROS
    ros::NodeHandle nh_private;
    ros::Publisher m_mapPub;
    ros::Publisher m_fullMapPub;
    ros::Publisher m_markerPub; 
    ros::Publisher m_pointCloudPub;

    ros::Publisher m_gridmapPub;

    tf::TransformListener tf_listener;

    dynamic_reconfigure::Server<covins_backend::GridMapperConfig> m_reconfigureServer;
    bool m_initConfig;
    boost::recursive_mutex m_config_mutex;


    protected:
    inline static void updateMinKey(const octomap::OcTreeKey& in, octomap::OcTreeKey& min) {
        for (unsigned i = 0; i < 3; ++i)
        min[i] = std::min(in[i], min[i]);
    };

    inline static void updateMaxKey(const octomap::OcTreeKey& in, octomap::OcTreeKey& max) {
        for (unsigned i = 0; i < 3; ++i)
        max[i] = std::max(in[i], max[i]);
    };

    /// Test if key is within update area of map (2D, ignores height)
    inline bool isInUpdateBBX(const OcTreeT::iterator& it) const {
        // 2^(tree_depth-depth) voxels wide:
        unsigned voxelWidth = (1 << (m_maxTreeDepth - it.getDepth()));
        octomap::OcTreeKey key = it.getIndexKey(); // lower corner of voxel
        return (key[0] + voxelWidth >= m_updateBBXMin[0]
                && key[1] + voxelWidth >= m_updateBBXMin[1]
                && key[0] <= m_updateBBXMax[0]
                && key[1] <= m_updateBBXMax[1]);
    }


    /**
     * @brief Find speckle nodes (single occupied voxels with no neighbors). Only works on lowest resolution!
     * @param key
     * @return
     */
    bool isSpeckleNode(const octomap::OcTreeKey& key) const;

    /// hook that is called before traversing all nodes
    virtual void handlePreNodeTraversal(const ros::Time& rostime);

    /// hook that is called when traversing all nodes of the updated Octree (does nothing here)
    virtual void handleNode(const OcTreeT::iterator& it) {};

    /// hook that is called when traversing all nodes of the updated Octree in the updated area (does nothing here)
    virtual void handleNodeInBBX(const OcTreeT::iterator& it) {};

    /// hook that is called when traversing occupied nodes of the updated Octree
    virtual void handleOccupiedNode(const OcTreeT::iterator& it);

    /// hook that is called when traversing occupied nodes in the updated area (updates 2D map projection here)
    virtual void handleOccupiedNodeInBBX(const OcTreeT::iterator& it);

    /// hook that is called when traversing free nodes of the updated Octree
    virtual void handleFreeNode(const OcTreeT::iterator& it);

    /// hook that is called when traversing free nodes in the updated area (updates 2D map projection here)
    virtual void handleFreeNodeInBBX(const OcTreeT::iterator& it);

    /// hook that is called after traversing all nodes
    virtual void handlePostNodeTraversal(const ros::Time& rostime);

    /// updates the downprojected 2D map as either occupied or free
    virtual void update2DMap(const OcTreeT::iterator& it, bool occupied);

    inline unsigned mapIdx(int i, int j) const {
        return m_gridmap.info.width * j + i;
    }

    inline unsigned mapIdx(const octomap::OcTreeKey& key) const {
        return mapIdx((key[0] - m_paddedMinKey[0]) / m_multires2DScale,
                    (key[1] - m_paddedMinKey[1]) / m_multires2DScale);

    }

    inline bool mapChanged(const nav_msgs::MapMetaData& oldMapInfo, const nav_msgs::MapMetaData& newMapInfo) {
        return (    oldMapInfo.height != newMapInfo.height
                || oldMapInfo.width != newMapInfo.width
                || oldMapInfo.origin.position.x != newMapInfo.origin.position.x
                || oldMapInfo.origin.position.y != newMapInfo.origin.position.y);
    }


};