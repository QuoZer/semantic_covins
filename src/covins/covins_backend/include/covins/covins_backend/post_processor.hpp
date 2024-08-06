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

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// ROS
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/OccupancyGrid.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <dynamic_reconfigure/server.h>
#include <covins_backend/GridMapperConfig.h>

// COVINS
#include "covins_backend/visualization_be.hpp"
#include "covins_backend/keyframe_be.hpp"
#include "covins_backend/landmark_be.hpp"

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

class MapPostProcessor {
public:
    MapPostProcessor(const ros::NodeHandle private_nh_ = ros::NodeHandle("~"));
    ~MapPostProcessor() = default;

    void reconfigureCallback(covins_backend::GridMapperConfig& config, uint32_t level);
    void axisDirCallback(const geometry_msgs::PoseWithCovarianceStamped& wall_axis);
    void updateTransform();


    pcl::PointCloud<pcl::PointXYZL>::Ptr createLabeledCloud( covins::VisualizerBase::LandmarkMap &landmarks );
    pcl::PointIndices::Ptr removeFurnitureNoise( const pcl::PointCloud<pcl::PointXYZL>::Ptr &cloud, std::list<int> &lm_ids,
        const std::vector<int> &obstacleLabels, const std::vector<int> &ignoreLabels, float clusterTolerance, int minClusterSize, int maxClusterSize);


    Eigen::Matrix3d axisRotMatrix;

    // ROS
    ros::NodeHandle nh_private;
    ros::Publisher m_mapPub;
    ros::Publisher m_fullMapPub;
    ros::Publisher m_markerPub; 
    ros::Publisher m_pointCloudPub;

    ros::Publisher m_gridmapPub;
    ros::Publisher m_statePub;


    tf::TransformListener tf_listener;

    dynamic_reconfigure::Server<covins_backend::GridMapperConfig> m_reconfigureServer;
    bool m_initConfig;
    boost::recursive_mutex m_config_mutex;

};