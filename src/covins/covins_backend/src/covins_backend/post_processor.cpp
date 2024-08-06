#include "covins_backend/post_processor.hpp"


bool is_equal (double a, double b, double epsilon = 1.0e-7)
{
    return std::abs(a - b) < epsilon;
}

MapPostProcessor::MapPostProcessor(const ros::NodeHandle private_nh_) 
    : nh_private(private_nh_)
      
{

    // m_fullMapPub = nh_private.advertise<octomap_msgs::Octomap>("octomap_full", 1, true);
    // m_gridmapPub = nh_private.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);
    m_markerPub = nh_private.advertise<visualization_msgs::MarkerArray>("occupied_cells_vis_array", 1, true);
    m_pointCloudPub = nh_private.advertise<sensor_msgs::PointCloud2>("covins_cloud_obstacles", 1, true);

    m_statePub = nh_private.advertise<std_msgs::Bool>("processing_state", 1, true);

    // axisSub = nh_private.subscribe("/initialpose", 5, &GridMapper::axisDirCallback, this);
    axisRotMatrix = Eigen::Matrix3d::Identity();

    dynamic_reconfigure::Server<covins_backend::GridMapperConfig>::CallbackType f;
    f = boost::bind(&MapPostProcessor::reconfigureCallback, this, boost::placeholders::_1, boost::placeholders::_2);
    m_reconfigureServer.setCallback(f);
}

void MapPostProcessor::reconfigureCallback(covins_backend::GridMapperConfig& config, uint32_t level){
    // ROS_INFO("Updating octomap params...");
//     if (m_maxTreeDepth != unsigned(config.max_depth))
//         m_maxTreeDepth = unsigned(config.max_depth);
//     else{

//     // Parameters with a namespace require an special treatment at the beginning, as dynamic reconfigure
//     // will overwrite them because the server is not able to match parameters' names.
//         if (m_initConfig){
//             // If parameters do not have the default value, dynamic reconfigure server should be updated.
//             if(!is_equal(m_maxRange, -1.0))
//                 config.sensor_model_max_range = m_maxRange;
//             if(!is_equal(m_minRange, -1.0))
//                 config.sensor_model_min_range = m_minRange;
//             if(!is_equal(m_octree->getProbHit(), 0.7))
//                 config.sensor_model_hit = m_octree->getProbHit();
//             if(!is_equal(m_octree->getProbMiss(), 0.4))
//                 config.sensor_model_miss = m_octree->getProbMiss();
//             if(!is_equal(m_octree->getClampingThresMin(), 0.12))
//                 config.sensor_model_min = m_octree->getClampingThresMin();
//             if(!is_equal(m_octree->getClampingThresMax(), 0.97))
//                 config.sensor_model_max = m_octree->getClampingThresMax();
//             m_initConfig = false;

//             boost::recursive_mutex::scoped_lock reconf_lock(m_config_mutex);
//             m_reconfigureServer.updateConfig(config);
//         }
//         else{
//             // m_maxRange                  = config.sensor_model_max_range;
//             // m_octree->setClampingThresMin(config.sensor_model_min);
//             // m_octree->setClampingThresMax(config.sensor_model_max);
//             // m_octree->setOccupancyThres(config.occupancy_threshold);

//             // // Checking values that might create unexpected behaviors.
//             // if (is_equal(config.sensor_model_hit, 1.0))
//             //     config.sensor_model_hit -= 1.0e-6;
//             // m_octree->setProbHit(config.sensor_model_hit);
//             // if (is_equal(config.sensor_model_miss, 0.0))
//             //     config.sensor_model_miss += 1.0e-6;
//             // m_octree->setProbMiss(config.sensor_model_miss);
//         }
//   }
}


void MapPostProcessor::updateTransform()
{
    tf::StampedTransform transform;
    try{    // will screw up if the tf changes mid map generation
      tf_listener.lookupTransform("/world", "/odom",
                               ros::Time(0), transform);
    }
    catch (tf::TransformException &ex) {
      ROS_ERROR("%s",ex.what());
      return;
    }
    // Convert the transform to an Eigen rotation matrix
    Eigen::Matrix3d rotation;
    tf::Quaternion quaternion = transform.getRotation();
    tf::Matrix3x3 tf_rotation(quaternion);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            rotation(i, j) = tf_rotation[i][j];
        }
    }

    axisRotMatrix = rotation; 
}


pcl::PointCloud<pcl::PointXYZL>::Ptr MapPostProcessor::createLabeledCloud( covins::VisualizerBase::LandmarkMap &landmarks )
{
    pcl::PointCloud<pcl::PointXYZL>::Ptr labeledCloud(new pcl::PointCloud<pcl::PointXYZL>);

    labeledCloud->width = 0;
    labeledCloud->height = 1;


    for(auto mit=landmarks.begin(); mit!=landmarks.end(); ++mit) {
        covins::VisualizerBase::LandmarkPtr lm_i = mit->second;
        Eigen::Vector3d PosWorld = lm_i->GetWorldPos();

        int class_label = covins::Visualizer::AggregateLabels(lm_i);
        pcl::PointXYZRGB p;
        pcl::PointXYZL np;
        p = covins::VisualizerBase::CreateSemanticPoint3D(PosWorld,lm_i->id_.second, class_label);
        np.x = p.x;
        np.y = p.y;
        np.z = p.z;
        np.label = class_label;

        labeledCloud->points.push_back(np);
        labeledCloud->width++;
        // removing the floor and ceiling lms
        // if (class_label == 5 || class_label == 3)
        //     continue; 
    }    

    return labeledCloud; 
}

pcl::PointIndices::Ptr MapPostProcessor::removeFurnitureNoise(const pcl::PointCloud<pcl::PointXYZL>::Ptr &cloud, std::list<int> &lm_ids,
                    const std::vector<int> &obstacleLabels, const std::vector<int> &ignoreLabels, float clusterTolerance, int minClusterSize, int maxClusterSize)
{
    pcl::PointCloud<pcl::PointXYZL>::Ptr rawObstacleCloud(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointIndices::Ptr rawObstacleIndices(new pcl::PointIndices);

    // Separate the obstacle classes into a separate pointcloud 
    int erased_lms = 0;
    for (int i = 0; i < cloud->size(); ++i) {
        const pcl::PointXYZL& point = cloud->points[i];
        int p_label = point.label; 
        if (std::find(obstacleLabels.begin(), obstacleLabels.end(), p_label) != obstacleLabels.end())  {
            rawObstacleIndices->indices.push_back(i);
        }
        else { // erase non-obsticles 
            if (i - erased_lms >= lm_ids.size()) {
                std::cout << "Landmark ids and ptc size don't match: " << i- erased_lms << " / " << lm_ids.size() << "\n";
                break;
            }
            std::list<int>::iterator it = lm_ids.begin();
            std::advance(it, i - erased_lms);
            lm_ids.erase(it);
            erased_lms++;
        }
    }

    std::cout << "Obstacle lms: " << lm_ids.size() << "\n";

    pcl::ExtractIndices<pcl::PointXYZL> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(rawObstacleIndices);
    extract.filter(*rawObstacleCloud);

    // save for debug
    pcl::io::savePCDFileASCII("/home/appuser/COVINS_demo/raw_cloud.pcd", *rawObstacleCloud);   

    pcl::PointCloud<pcl::PointXYZL>::Ptr obstacleCloud(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointIndices::Ptr obstacleIndices(new pcl::PointIndices);

    // Create a KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZL>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZL>);

    for (int label = 0; label <150; label++) {
        
        // skip things we dont want to use for clustering 
        if ( (std::find(obstacleLabels.begin(), obstacleLabels.end(), label) != obstacleLabels.end()) || 
             (std::find(ignoreLabels.begin(), ignoreLabels.end(), label) != ignoreLabels.end()) )
             continue;

        pcl::PointCloud<pcl::PointXYZL>::Ptr furnitureCloud(new pcl::PointCloud<pcl::PointXYZL>);
        pcl::PointIndices::Ptr indices(new pcl::PointIndices);

        // Separate points with the current class label
        for (int i = 0; i < cloud->size(); ++i) {
            if (cloud->points[i].label == label) {
                indices->indices.push_back(i);
            }
        }

        if (indices->indices.size() == 0) continue; 

        pcl::ExtractIndices<pcl::PointXYZL> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(indices);
        extract.filter(*furnitureCloud);

        // Perform DBSCAN clustering
        tree->setInputCloud(furnitureCloud);
        std::vector<pcl::PointIndices> clusterIndices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZL> ec;
        ec.setClusterTolerance(clusterTolerance);
        ec.setMinClusterSize(minClusterSize);
        ec.setMaxClusterSize(maxClusterSize);
        ec.setSearchMethod(tree);
        ec.setInputCloud(furnitureCloud);
        ec.extract(clusterIndices);

        std::cout << "For class " << label << " extracted " << clusterIndices.size() << " clusters \n";

        // Calculate bounding boxes for each cluster and remove obstacle points inside them
        for (const pcl::PointIndices& cluster : clusterIndices) {
            pcl::PointCloud<pcl::PointXYZL>::Ptr clusterCloud(new pcl::PointCloud<pcl::PointXYZL>);
            for (const int index : cluster.indices) {
                clusterCloud->points.push_back(furnitureCloud->points[index]);
            }
            clusterCloud->width = clusterCloud->points.size();
            clusterCloud->height = 1;
            clusterCloud->is_dense = true;

            Eigen::Vector4f minPoint, maxPoint;
            pcl::getMinMax3D(*clusterCloud, minPoint, maxPoint);

            for (int i = 0; i < rawObstacleCloud->size(); ++i) {
                const pcl::PointXYZL& point = rawObstacleCloud->points[i];
                if (point.label != label &&
                    point.x >= minPoint[0] && point.x <= maxPoint[0] &&
                    point.y >= minPoint[1] && point.y <= maxPoint[1] &&
                    point.z >= minPoint[2] && point.z <= maxPoint[2]) {
                    obstacleIndices->indices.push_back(i);
                }
            }
        }
    }

    std::cout << "Removed " << obstacleIndices->indices.size() << " points \n";

    // Extract the remaining obstacle points
    pcl::ExtractIndices<pcl::PointXYZL> extractnoise;
    extractnoise.setInputCloud(rawObstacleCloud);
    extractnoise.setIndices(obstacleIndices);
    extractnoise.setNegative(true);
    extractnoise.filter(*obstacleCloud);

    int erased = 0;
    for (int index : obstacleIndices->indices) {
        std::list<int>::iterator it = lm_ids.begin();
        std::advance(it, index - erased);
        lm_ids.erase(it);
        erased++;
    }

    std::cout << "Saving pointclouds... \n";

    // save for debug
    pcl::io::savePCDFileASCII("/home/appuser/COVINS_demo/clean_cloud.pcd", *obstacleCloud);   

    extractnoise.setNegative(false);
    extractnoise.filter(*obstacleCloud);

    // save for debug
    pcl::io::savePCDFileASCII("/home/appuser/COVINS_demo/noise_points.pcd", *obstacleCloud);  

    return obstacleIndices;
}
void MapPostProcessor::axisDirCallback(const geometry_msgs::PoseWithCovarianceStamped &wall_axis)
{
    // Extract quaternion from message
    geometry_msgs::Quaternion quat = wall_axis.pose.pose.orientation;

    // Convert quaternion to Eigen rotation matrix
    Eigen::Quaterniond eigenQuat(quat.w, quat.x, quat.y, quat.z);
    axisRotMatrix = eigenQuat.normalized().toRotationMatrix();
    
}
