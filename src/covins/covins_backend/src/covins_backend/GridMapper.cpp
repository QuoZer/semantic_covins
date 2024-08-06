#include "covins_backend/GridMapper.hpp"


bool is_equal (double a, double b, double epsilon = 1.0e-7)
{
    return std::abs(a - b) < epsilon;
}

GridMapper::GridMapper(const ros::NodeHandle private_nh_) 
    : nh_private(private_nh_),
      oc_resolution(0.1),
      m_worldFrameId("odom"),
      m_minRange(-1.0),
      m_maxRange(10.0),
      m_publish2DMap(false),
      m_initConfig(true),
      m_useColoredMap(false)     
      
{
    map = grid_map::GridMap({"occupancy", "elevation", "objects"});
    // map.setFrameId("map");
    // map.setGeometry(Length(10, 10), 0.1, Position(0.0,0.0));
    // ROS_INFO("Created map with size %f x %f m (%i x %i cells).\n The center of the map is located at (%f, %f) in the %s frame.",
    //     map.getLength().x(), map.getLength().y(),
    //     map.getSize()(0), map.getSize()(1),
    //     map.getPosition().x(), map.getPosition().y(), map.getFrameId().c_str());

    // params
    double probHit, probMiss, thresMin, thresMax;
    nh_private.param("resolution", oc_resolution, oc_resolution);
    nh_private.param("sensor_model/hit", probHit, 0.7);
    nh_private.param("sensor_model/miss", probMiss, 0.49);
    nh_private.param("sensor_model/min", thresMin, 0.1);
    nh_private.param("sensor_model/max", thresMax, 0.90);

    m_fullMapPub = nh_private.advertise<octomap_msgs::Octomap>("octomap_full", 1, true);
    m_gridmapPub = nh_private.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);
    m_markerPub = nh_private.advertise<visualization_msgs::MarkerArray>("occupied_cells_vis_array", 1, true);
    m_pointCloudPub = nh_private.advertise<sensor_msgs::PointCloud2>("covins_cloud_obstacles", 1, true);

    // axisSub = nh_private.subscribe("/initialpose", 5, &GridMapper::axisDirCallback, this);
    axisRotMatrix = Eigen::Matrix3d::Identity();

    // initialize octomap object & params
    m_octree = new OcTreeT(oc_resolution);
    m_octree->setProbHit(probHit);
    m_octree->setProbMiss(probMiss);
    m_octree->setClampingThresMin(thresMin);
    m_octree->setClampingThresMax(thresMax);
    m_octree->setOccupancyThres(0.5);
    m_treeDepth = m_octree->getTreeDepth();
    m_maxTreeDepth = m_treeDepth;
    m_gridmap.info.resolution = oc_resolution;

    dynamic_reconfigure::Server<covins_backend::GridMapperConfig>::CallbackType f;
    f = boost::bind(&GridMapper::reconfigureCallback, this, boost::placeholders::_1, boost::placeholders::_2);
    m_reconfigureServer.setCallback(f);

    ROS_INFO("OcTree created ");
}

void GridMapper::reconfigureCallback(covins_backend::GridMapperConfig& config, uint32_t level){
    ROS_INFO("Updating octomap params...");
    if (m_maxTreeDepth != unsigned(config.max_depth))
        m_maxTreeDepth = unsigned(config.max_depth);
    else{
    // m_pointcloudMinZ            = config.pointcloud_min_z;
    // m_pointcloudMaxZ            = config.pointcloud_max_z;
    // m_occupancyMinZ             = config.occupancy_min_z;
    // // m_occupancyMaxZ             = config.occupancy_max_z;
    // m_filterSpeckles            = config.filter_speckles;
    // m_filterGroundPlane         = config.filter_ground;
    // m_compressMap               = config.compress_map;
    // m_incrementalUpdate         = config.incremental_2D_projection;

    // Parameters with a namespace require an special treatment at the beginning, as dynamic reconfigure
    // will overwrite them because the server is not able to match parameters' names.
        if (m_initConfig){
            // If parameters do not have the default value, dynamic reconfigure server should be updated.
            if(!is_equal(m_maxRange, -1.0))
                config.sensor_model_max_range = m_maxRange;
            if(!is_equal(m_minRange, -1.0))
                config.sensor_model_min_range = m_minRange;
            if(!is_equal(m_octree->getProbHit(), 0.7))
                config.sensor_model_hit = m_octree->getProbHit();
            if(!is_equal(m_octree->getProbMiss(), 0.4))
                config.sensor_model_miss = m_octree->getProbMiss();
            if(!is_equal(m_octree->getClampingThresMin(), 0.12))
                config.sensor_model_min = m_octree->getClampingThresMin();
            if(!is_equal(m_octree->getClampingThresMax(), 0.97))
                config.sensor_model_max = m_octree->getClampingThresMax();
            m_initConfig = false;

            boost::recursive_mutex::scoped_lock reconf_lock(m_config_mutex);
            m_reconfigureServer.updateConfig(config);
        }
        else{
            m_maxRange                  = config.sensor_model_max_range;
            m_octree->setClampingThresMin(config.sensor_model_min);
            m_octree->setClampingThresMax(config.sensor_model_max);
            m_octree->setOccupancyThres(config.occupancy_threshold);

            // Checking values that might create unexpected behaviors.
            if (is_equal(config.sensor_model_hit, 1.0))
                config.sensor_model_hit -= 1.0e-6;
            m_octree->setProbHit(config.sensor_model_hit);
            if (is_equal(config.sensor_model_miss, 0.0))
                config.sensor_model_miss += 1.0e-6;
            m_octree->setProbMiss(config.sensor_model_miss);
        }
  }
}

void GridMapper::resetMap()
{
    m_octree->clear();
    map.clearAll();
}

void GridMapper::updateTransform()
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

void GridMapper::printTreeParams()
{
    std::cout << "OcTree params: " << 
                 "\n depth " << m_octree->getTreeDepth() << 
                 "\n resolution " << m_octree->getResolution() <<
                 "\n hit prob " << m_octree->getProbHit() <<
                 "\n miss prob " << m_octree->getProbMiss() <<
                 "\n max distance " << m_maxRange << 
                 "\n occupancy threshold " << m_octree->getOccupancyThres() << std::endl;
}

void GridMapper::insertKF(covins::VisualizerBase::KeyframePtr kf)
{
    // std::cout << "INSERTING KF \n"; 
    covins::TypeDefs::TransformType camera_origin = kf->GetPoseTws();
    covins::TypeDefs::Vector3Type rot_camera = axisRotMatrix * camera_origin.block<3,1>(0, 3);
    point3d camera_pos_om = point3d(rot_camera(0), rot_camera(1), rot_camera(2));
    // std::cout << "GOT KF POSE \n"; 
    if (!m_octree->coordToKeyChecked(camera_pos_om, m_updateBBXMin)
        || !m_octree->coordToKeyChecked(camera_pos_om, m_updateBBXMax))
    {
        ROS_ERROR_STREAM("Could not generate Key for camera "<< camera_pos_om);
    }
    // get the relevant landmarks
    covins::TypeDefs::LandmarkVector landmarks = kf->GetValidLandmarks();

    int skipped_lms = 0;
    // ground, ceiling, lights (+ mats and some other classes) are not considered obstacles (table id-1)
    std::vector<int> non_obstacle_classes = {3, 5, 82}; // 
    std::vector<int> obstacle_classes = {0}; // only walls
    
    KeySet free_cells, occupied_cells;
    std::vector<int> cell_classes;
    for (covins::TypeDefs::LandmarkPtr lm : landmarks)
    {
        // get the landmark position
        Eigen::Matrix<double,3,1> lm_pos = lm->GetWorldPos();
        covins::TypeDefs::Vector3Type rot_lm = axisRotMatrix * lm_pos;
        point3d lm_pos_om = point3d(rot_lm(0), rot_lm(1), rot_lm(2));
        // std::cout << "GOT LM POS \n"; 
        
        // too close
        if ((m_minRange > 0) && ((lm_pos_om - camera_pos_om).norm() < m_minRange) ) {
            skipped_lms++;
            continue;
        }
        
        int lm_class =  covins::Visualizer::AggregateLabels(lm);        // cachable
        
        // bool is_obstacle = std::find(non_obstacle_classes.begin(), non_obstacle_classes.end(), lm_class) == non_obstacle_classes.end();
        // bool is_obstacle = std::find(obstacle_classes.begin(), obstacle_classes.end(), lm_class) != obstacle_classes.end();
        // bool is_obstacle = true ? lm_class == 0 : false; // just the walls 
        bool is_obstacle = true;

        // too far 
        if ( (m_maxRange > 0.0) && ( (lm_pos_om - camera_pos_om).norm() >= m_maxRange) )
        {
            skipped_lms++;
            continue;       // or truncate:
            lm_pos_om = camera_pos_om + (lm_pos_om - camera_pos_om).normalized() * m_maxRange; 
            is_obstacle = false; 
        }
        // get the ray from camera to landmark
        if (m_octree->computeRayKeys(camera_pos_om, lm_pos_om, m_keyRay))
        {
            free_cells.insert(m_keyRay.begin(), m_keyRay.end());
        }   
        // map walls at lower resolution
        // int node_depth = lm_class == 0  ? 15 : 16;
        // std::cout << "Node class " << lm_class << " node depth " << node_depth << "\n";
        octomap::OcTreeKey endKey;  // 
        if (m_octree->coordToKeyChecked(lm_pos_om, endKey)){
            updateMinKey(endKey, m_updateBBXMin);
            updateMaxKey(endKey, m_updateBBXMax);

            if (is_obstacle){
                occupied_cells.insert(endKey);
                cell_classes.push_back(lm_class);
            }
        }

    }
    // ROS_INFO_STREAM("Skipped: " << skipped_lms << "/" << landmarks.size() << " lms");

    // mark free cells only if not seen occupied in this cloud
    for(KeySet::iterator it = free_cells.begin(), end=free_cells.end(); it!= end; ++it){
        if (occupied_cells.find(*it) == occupied_cells.end()){
            m_octree->updateNode(*it, false);
        }
    }
    int cell_id = 0;
    // now mark all occupied cells:
    for (KeySet::iterator it = occupied_cells.begin(), end=occupied_cells.end(); it!= end; it++) {
        m_octree->updateNode(*it, true);
        int class_label = cell_classes.at(cell_id);
        // Store the class in r-channel
        m_octree->setNodeColor(*it, covins_params::ade20k::class_colors[class_label].mu8R,
                                    covins_params::ade20k::class_colors[class_label].mu8G,
                                    covins_params::ade20k::class_colors[class_label].mu8B);
        cell_id++;
        // m_octree->averageNodeColor(*it, covins_params::ade20k::class_colors[class_label].mu8R,
        //                             covins_params::ade20k::class_colors[class_label].mu8G,
        //                             covins_params::ade20k::class_colors[class_label].mu8B);
        // std::cout << "Assigned class: " << static_cast<int>(r) << std::endl;
    }

    // prune?
    // m_octree->prune();

}

pcl::PointCloud<pcl::PointXYZL>::Ptr GridMapper::createLabeledCloud( covins::VisualizerBase::LandmarkMap &landmarks )
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

pcl::PointIndices::Ptr GridMapper::removeFurnitureNoise(const pcl::PointCloud<pcl::PointXYZL>::Ptr &cloud, std::list<int> &lm_ids,
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
void GridMapper::axisDirCallback(const geometry_msgs::PoseWithCovarianceStamped &wall_axis)
{
    // Extract quaternion from message
    geometry_msgs::Quaternion quat = wall_axis.pose.pose.orientation;

    // Convert quaternion to Eigen rotation matrix
    Eigen::Quaterniond eigenQuat(quat.w, quat.x, quat.y, quat.z);
    axisRotMatrix = eigenQuat.normalized().toRotationMatrix();
    
}


void GridMapper::publishFullOctoMap(const ros::Time& rostime) const
{
    octomap_msgs::Octomap oc_map;
    oc_map.header.frame_id = m_worldFrameId;
    oc_map.header.stamp = rostime;

    if (octomap_msgs::fullMapToMsg(*m_octree, oc_map))
        m_fullMapPub.publish(oc_map);
    else
        ROS_ERROR("Error serializing OctoMap");

    m_octree->write("/home/appuser/COVINS_demo/src/covins/covins_backend/output/last_octomap.ot");

}

void GridMapper::publishBinaryOctoMap(const ros::Time& rostime) const
{
    octomap_msgs::Octomap oc_map;
    oc_map.header.frame_id = m_worldFrameId;
    oc_map.header.stamp = rostime;

    if (octomap_msgs::binaryMapToMsg(*m_octree, oc_map))
        m_fullMapPub.publish(oc_map);
    else
        ROS_ERROR("Error serializing OctoMap");

}

void GridMapper::publishMapAsMarkers(const ros::Time& rostime, bool publishMarkerArray)
{
    bool publishPointCloud =  !publishMarkerArray;
    bool publishFreeMarkerArray = false; 
    bool m_publishFreeSpace = false; 
    bool m_filterSpeckles = false; 

    ros::WallTime startTime = ros::WallTime::now();
    size_t octomapSize = m_octree->size();
    // TODO: estimate num occ. voxels for size of arrays (reserve)
    if (octomapSize <= 1){
        ROS_WARN("Nothing to publish, octree is empty");
        return;
    }

    // init markers for free space:
    visualization_msgs::MarkerArray freeNodesVis;
    // init pointcloud:
    pcl::PointCloud<pcl::PointXYZRGB> pclCloud;
    // each array stores all cubes of a different size, one for each depth level:
    freeNodesVis.markers.resize(m_treeDepth+1);

    geometry_msgs::Pose pose;
    pose.orientation = tf::createQuaternionMsgFromYaw(0.0);

    // init markers:
    visualization_msgs::MarkerArray occupiedNodesVis;
    // each array stores all cubes of a different size, one for each depth level:
    occupiedNodesVis.markers.resize(m_treeDepth+1);

    // call pre-traversal hook:
    handlePreNodeTraversal(rostime);

    // now, traverse all leafs in the tree:
    for (OcTreeT::iterator it = m_octree->begin(m_maxTreeDepth),
        end = m_octree->end(); it != end; ++it)
    {
        bool inUpdateBBX = isInUpdateBBX(it);

        // call general hook:
        handleNode(it);
        if (inUpdateBBX)
            handleNodeInBBX(it);

        if (m_octree->isNodeOccupied(*it)){
            double z = it.getZ();
            double half_size = it.getSize() / 2.0;
            // if (z + half_size > m_occupancyMinZ && z - half_size < m_occupancyMaxZ)
            if (true)
            {
                double size = it.getSize();
                double x = it.getX();
                double y = it.getY();
                int r = it->getColor().r;
                int g = it->getColor().g;
                int b = it->getColor().b;

                // Ignore speckles in the map:
                if (m_filterSpeckles && (it.getDepth() == m_treeDepth +1) && isSpeckleNode(it.getKey())){
                    ROS_DEBUG("Ignoring single speckle at (%f,%f,%f)", x, y, z);
                    continue;
                } // else: current octree node is no speckle, send it out

                handleOccupiedNode(it);
                if (inUpdateBBX)
                    handleOccupiedNodeInBBX(it);


                //create marker:
                if (publishMarkerArray){
                    unsigned idx = it.getDepth();
                    assert(idx < occupiedNodesVis.markers.size());

                    geometry_msgs::Point cubeCenter;
                    cubeCenter.x = x;
                    cubeCenter.y = y;
                    cubeCenter.z = z;

                    occupiedNodesVis.markers[idx].points.push_back(cubeCenter);
                    // if (m_useHeightMap){
                    //     double minX, minY, minZ, maxX, maxY, maxZ;
                    //     m_octree->getMetricMin(minX, minY, minZ);
                    //     m_octree->getMetricMax(maxX, maxY, maxZ);

                    //     double h = (1.0 - std::min(std::max((cubeCenter.z-minZ)/ (maxZ - minZ), 0.0), 1.0)) *m_colorFactor;
                    //     occupiedNodesVis.markers[idx].colors.push_back(heightMapColor(h));
                    // }

                #ifdef COLOR_OCTOMAP_SERVER
                    if (m_useColoredMap) {
                        std_msgs::ColorRGBA _color; _color.r = (r / 255.); _color.g = (g / 255.); _color.b = (b / 255.); _color.a = 1.0; // TODO/EVALUATE: potentially use occupancy as measure for alpha channel?
                        occupiedNodesVis.markers[idx].colors.push_back(_color);
                    }
                #endif
                }
                        // insert into pointcloud:
                    if (publishPointCloud) {
                        pcl::PointXYZRGB _point = pcl::PointXYZRGB();
                        _point.x = x; _point.y = y; _point.z = z;
                        _point.r = r; _point.g = g; _point.b = b;
                        pclCloud.push_back(_point);
                    }
            }
        } else{ // node not occupied => mark as free in 2D map if unknown so far
            double z = it.getZ();
            double half_size = it.getSize() / 2.0;
            // if (z + half_size > m_occupancyMinZ && z - half_size < m_occupancyMaxZ)
            if (true)
            {
                handleFreeNode(it);
                if (inUpdateBBX)
                    handleFreeNodeInBBX(it);

                if (m_publishFreeSpace){
                    double x = it.getX();
                    double y = it.getY();

                    //create marker for free space:
                    unsigned idx = it.getDepth();
                    assert(idx < freeNodesVis.markers.size());

                    geometry_msgs::Point cubeCenter;
                    cubeCenter.x = x;
                    cubeCenter.y = y;
                    cubeCenter.z = z;

                    freeNodesVis.markers[idx].points.push_back(cubeCenter);
                    
                }
            }
        }
    }

    // call post-traversal hook:
    // handlePostNodeTraversal(rostime);

    // finish MarkerArray:
    if (publishMarkerArray){
        for (unsigned i= 0; i < occupiedNodesVis.markers.size(); ++i){
            double size = m_octree->getNodeSize(i);

            occupiedNodesVis.markers[i].header.frame_id = m_worldFrameId;
            occupiedNodesVis.markers[i].header.stamp = rostime;
            occupiedNodesVis.markers[i].ns = "map";
            occupiedNodesVis.markers[i].id = i;
            occupiedNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
            occupiedNodesVis.markers[i].scale.x = size;
            occupiedNodesVis.markers[i].scale.y = size;
            occupiedNodesVis.markers[i].scale.z = size;
            occupiedNodesVis.markers[i].pose.orientation.x=0;
            occupiedNodesVis.markers[i].pose.orientation.y=0;
            occupiedNodesVis.markers[i].pose.orientation.z=0;
            occupiedNodesVis.markers[i].pose.orientation.w=1;
            // if (!m_useColoredMap)
            //     occupiedNodesVis.markers[i].color = m_color;  HACK 


            if (occupiedNodesVis.markers[i].points.size() > 0)
                occupiedNodesVis.markers[i].action = visualization_msgs::Marker::ADD;
            else
                occupiedNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
        }

        m_markerPub.publish(occupiedNodesVis);
    }


    // finish FreeMarkerArray:
    if (publishFreeMarkerArray){
        for (unsigned i= 0; i < freeNodesVis.markers.size(); ++i){
            double size = m_octree->getNodeSize(i);

            freeNodesVis.markers[i].header.frame_id = m_worldFrameId;
            freeNodesVis.markers[i].header.stamp = rostime;
            freeNodesVis.markers[i].ns = "map";
            freeNodesVis.markers[i].id = i;
            freeNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
            freeNodesVis.markers[i].scale.x = size;
            freeNodesVis.markers[i].scale.y = size;
            freeNodesVis.markers[i].scale.z = size;
            // freeNodesVis.markers[i].color = m_colorFree;  HACK 


            if (freeNodesVis.markers[i].points.size() > 0)
                freeNodesVis.markers[i].action = visualization_msgs::Marker::ADD;
            else
                freeNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
        }

        // m_fmarkerPub.publish(freeNodesVis);  // HACK
    }

    // finish pointcloud:
    if (publishPointCloud){
        sensor_msgs::PointCloud2 cloud;
        pcl::toROSMsg (pclCloud, cloud);
        cloud.header.frame_id = "odom";
        cloud.header.stamp = rostime;
        m_pointCloudPub.publish(cloud);
        pcl::io::savePCDFileASCII ("/home/appuser/COVINS_demo/src/covins/covins_backend/output/last_pcd.pcd", pclCloud);
    }


    double total_elapsed = (ros::WallTime::now() - startTime).toSec();
    ROS_INFO("Map publishing in GridMapper took %f sec", total_elapsed);
    
}

void GridMapper::handlePreNodeTraversal(const ros::Time& rostime){
    double m_minSizeX = 10;
    double m_minSizeY = 10;     // HACK
    if (m_publish2DMap){
        // init projected 2D map:
        m_gridmap.header.frame_id = m_worldFrameId;
        m_gridmap.header.stamp = rostime;
        nav_msgs::MapMetaData oldMapInfo = m_gridmap.info;

        // TODO: move most of this stuff into c'tor and init map only once (adjust if size changes)
        double minX, minY, minZ, maxX, maxY, maxZ;
        m_octree->getMetricMin(minX, minY, minZ);
        m_octree->getMetricMax(maxX, maxY, maxZ);

        octomap::point3d minPt(minX, minY, minZ);
        octomap::point3d maxPt(maxX, maxY, maxZ);
        octomap::OcTreeKey minKey = m_octree->coordToKey(minPt, m_maxTreeDepth);
        octomap::OcTreeKey maxKey = m_octree->coordToKey(maxPt, m_maxTreeDepth);

        ROS_DEBUG("MinKey: %d %d %d / MaxKey: %d %d %d", minKey[0], minKey[1], minKey[2], maxKey[0], maxKey[1], maxKey[2]);

        // add padding if requested (= new min/maxPts in x&y):
        double halfPaddedX = 0.5*m_minSizeX;
        double halfPaddedY = 0.5*m_minSizeY;
        minX = std::min(minX, -halfPaddedX);
        maxX = std::max(maxX, halfPaddedX);
        minY = std::min(minY, -halfPaddedY);
        maxY = std::max(maxY, halfPaddedY);
        minPt = octomap::point3d(minX, minY, minZ);
        maxPt = octomap::point3d(maxX, maxY, maxZ);

        OcTreeKey paddedMaxKey;
        if (!m_octree->coordToKeyChecked(minPt, m_maxTreeDepth, m_paddedMinKey)){
            ROS_ERROR("Could not create padded min OcTree key at %f %f %f", minPt.x(), minPt.y(), minPt.z());
            return;
        }
        if (!m_octree->coordToKeyChecked(maxPt, m_maxTreeDepth, paddedMaxKey)){
            ROS_ERROR("Could not create padded max OcTree key at %f %f %f", maxPt.x(), maxPt.y(), maxPt.z());
            return;
        }

        ROS_DEBUG("Padded MinKey: %d %d %d / padded MaxKey: %d %d %d", m_paddedMinKey[0], m_paddedMinKey[1], m_paddedMinKey[2], paddedMaxKey[0], paddedMaxKey[1], paddedMaxKey[2]);
        assert(paddedMaxKey[0] >= maxKey[0] && paddedMaxKey[1] >= maxKey[1]);

        m_multires2DScale = 1 << (m_treeDepth - m_maxTreeDepth);
        m_gridmap.info.width = (paddedMaxKey[0] - m_paddedMinKey[0])/m_multires2DScale +1;
        m_gridmap.info.height = (paddedMaxKey[1] - m_paddedMinKey[1])/m_multires2DScale +1;

        int mapOriginX = minKey[0] - m_paddedMinKey[0];
        int mapOriginY = minKey[1] - m_paddedMinKey[1];
        assert(mapOriginX >= 0 && mapOriginY >= 0);

        // might not exactly be min / max of octree:
        octomap::point3d origin = m_octree->keyToCoord(m_paddedMinKey, m_treeDepth);
        double gridRes = m_octree->getNodeSize(m_maxTreeDepth);
        m_projectCompleteMap = ( (std::abs(gridRes-m_gridmap.info.resolution) > 1e-6)); //(!m_incrementalUpdate || (std::abs(gridRes-m_gridmap.info.resolution) > 1e-6))
        m_gridmap.info.resolution = gridRes;
        m_gridmap.info.origin.position.x = origin.x() - gridRes*0.5;
        m_gridmap.info.origin.position.y = origin.y() - gridRes*0.5;
        if (m_maxTreeDepth != m_treeDepth){
            m_gridmap.info.origin.position.x -= oc_resolution/2.0;
            m_gridmap.info.origin.position.y -= oc_resolution/2.0;
        }

        // workaround for  multires. projection not working properly for inner nodes:
        // force re-building complete map
        bool m_projectCompleteMap = false;
        if (m_maxTreeDepth < m_treeDepth)
            m_projectCompleteMap = true;


        if(m_projectCompleteMap){
            ROS_DEBUG("Rebuilding complete 2D map");
            m_gridmap.data.clear();
            // init to unknown:
            m_gridmap.data.resize(m_gridmap.info.width * m_gridmap.info.height, -1);

        } else {

        if (mapChanged(oldMapInfo, m_gridmap.info)){
            ROS_DEBUG("2D grid map size changed to %dx%d", m_gridmap.info.width, m_gridmap.info.height);
            // HACK adjustMapData(m_gridmap, oldMapInfo);
        }
        nav_msgs::OccupancyGrid::_data_type::iterator startIt;
        size_t mapUpdateBBXMinX = std::max(0, (int(m_updateBBXMin[0]) - int(m_paddedMinKey[0]))/int(m_multires2DScale));
        size_t mapUpdateBBXMinY = std::max(0, (int(m_updateBBXMin[1]) - int(m_paddedMinKey[1]))/int(m_multires2DScale));
        size_t mapUpdateBBXMaxX = std::min(int(m_gridmap.info.width-1), (int(m_updateBBXMax[0]) - int(m_paddedMinKey[0]))/int(m_multires2DScale));
        size_t mapUpdateBBXMaxY = std::min(int(m_gridmap.info.height-1), (int(m_updateBBXMax[1]) - int(m_paddedMinKey[1]))/int(m_multires2DScale));

        assert(mapUpdateBBXMaxX > mapUpdateBBXMinX);
        assert(mapUpdateBBXMaxY > mapUpdateBBXMinY);

        size_t numCols = mapUpdateBBXMaxX-mapUpdateBBXMinX +1;

        // test for max idx:
        uint max_idx = m_gridmap.info.width*mapUpdateBBXMaxY + mapUpdateBBXMaxX;
        if (max_idx  >= m_gridmap.data.size())
            ROS_ERROR("BBX index not valid: %d (max index %zu for size %d x %d) update-BBX is: [%zu %zu]-[%zu %zu]", max_idx, m_gridmap.data.size(), m_gridmap.info.width, m_gridmap.info.height, mapUpdateBBXMinX, mapUpdateBBXMinY, mapUpdateBBXMaxX, mapUpdateBBXMaxY);

        // reset proj. 2D map in bounding box:
        for (unsigned int j = mapUpdateBBXMinY; j <= mapUpdateBBXMaxY; ++j){
            std::fill_n(m_gridmap.data.begin() + m_gridmap.info.width*j+mapUpdateBBXMinX,
                        numCols, -1);
        }

        }



    }

}

void GridMapper::handlePostNodeTraversal(const ros::Time& rostime){
  publishProjected2DMap(rostime);
}

void GridMapper::handleOccupiedNode(const OcTreeT::iterator& it){
  if (m_publish2DMap && m_projectCompleteMap){
    update2DMap(it, true);
  }
}

void GridMapper::handleFreeNode(const OcTreeT::iterator& it){
    bool m_projectCompleteMap = false; //HACK
  if (m_publish2DMap && m_projectCompleteMap){
    update2DMap(it, false);
  }
}

void GridMapper::handleOccupiedNodeInBBX(const OcTreeT::iterator& it){
  if (m_publish2DMap && !m_projectCompleteMap){
    update2DMap(it, true);
  }
}

void GridMapper::handleFreeNodeInBBX(const OcTreeT::iterator& it){
  if (m_publish2DMap && !m_projectCompleteMap){
    update2DMap(it, false);
  }
}

void GridMapper::update2DMap(const OcTreeT::iterator& it, bool occupied){
  // update 2D map (occupied always overrides):

    if (it.getDepth() == m_maxTreeDepth){
        unsigned idx = mapIdx(it.getKey());
        if (occupied)
            m_gridmap.data[mapIdx(it.getKey())] = 100;
        else if (m_gridmap.data[idx] == -1){
            m_gridmap.data[idx] = 0;
        }

    } else{
        int intSize = 1 << (m_maxTreeDepth - it.getDepth());
        octomap::OcTreeKey minKey=it.getIndexKey();
        for(int dx=0; dx < intSize; dx++){
            int i = (minKey[0]+dx - m_paddedMinKey[0])/m_multires2DScale;
            for(int dy=0; dy < intSize; dy++){
                unsigned idx = mapIdx(i, (minKey[1]+dy - m_paddedMinKey[1])/m_multires2DScale);
                if (occupied)
                    m_gridmap.data[idx] = 100;
                else if (m_gridmap.data[idx] == -1){
                    m_gridmap.data[idx] = 0;
                }
            }
        }
    }


}



bool GridMapper::isSpeckleNode(const OcTreeKey&nKey) const {
    OcTreeKey key;
    bool neighborFound = false;
    for (key[2] = nKey[2] - 1; !neighborFound && key[2] <= nKey[2] + 1; ++key[2]){
        for (key[1] = nKey[1] - 1; !neighborFound && key[1] <= nKey[1] + 1; ++key[1]){
        for (key[0] = nKey[0] - 1; !neighborFound && key[0] <= nKey[0] + 1; ++key[0]){
            if (key != nKey){
                OcTreeNode* node = m_octree->search(key);
            if (node && m_octree->isNodeOccupied(node)){
                // we have a neighbor => break!
                neighborFound = true;
            }
            }
        }
        }
    }

    return neighborFound;
}


bool GridMapper::populateGridMap()
{
    // Iterate through leaf nodes and project occupied cells to elevation map.
    // On the first pass, expand all occupied cells that are not at maximum depth.
    unsigned int max_depth = m_octree->getTreeDepth();
    // Adapted from octomap octree2pointcloud.cpp.
    std::vector<octomap::ColorOcTreeNode*> collapsed_occ_nodes;
    do {
        collapsed_occ_nodes.clear();
        for (octomap::ColorOcTree::iterator it = m_octree->begin(); it != m_octree->end(); ++it) {
        if (m_octree->isNodeOccupied(*it) && it.getDepth() < max_depth) {
            collapsed_occ_nodes.push_back(&(*it));
        }
        }
        for (std::vector<octomap::ColorOcTreeNode*>::iterator it = collapsed_occ_nodes.begin();
                                                it != collapsed_occ_nodes.end(); ++it) {
            // octomap::ColorOcTreeNode* node = dynamic_cast<octomap::ColorOcTreeNode*>(*it);
            
            m_octree->expandNode(*it);
        }
        std::cout << "Expanded " << collapsed_occ_nodes.size() << " nodes" << std::endl;
    } while (collapsed_occ_nodes.size() > 0 );

    // Set up grid map geometry.
    // TODO Figure out whether to center map.
    double resolution = m_octree->getResolution();
    grid_map::Position3 minBound;
    grid_map::Position3 maxBound;
    m_octree->getMetricMin(minBound(0), minBound(1), minBound(2));
    m_octree->getMetricMax(maxBound(0), maxBound(1), maxBound(2));
    // User can provide coordinate limits to only convert a bounding box.
    octomap::point3d minBbx(minBound(0), minBound(1), minBound(2));
    octomap::point3d maxBbx(maxBound(0), maxBound(1), maxBound(2));

    grid_map::Length length = grid_map::Length(maxBound(0) - minBound(0), maxBound(1) - minBound(1));
    grid_map::Position position = grid_map::Position((maxBound(0) + minBound(0)) / 2.0,
                                                    (maxBound(1) + minBound(1)) / 2.0);
    map.setGeometry(length, resolution, position);
    std::cout << "grid map geometry: " << std::endl;
    std::cout << "Length: [" << length(0) << ", " << length(1) << "]" << std::endl;
    std::cout << "Position: [" << position(0) << ", " << position(1) << "]" << std::endl;
    std::cout << "Resolution: " << resolution << std::endl;


    // std::cout << "Iterating from " << min_bbx << " to " << max_bbx << std::endl;
    grid_map::Matrix& gridMapData = map["occupancy"];
    grid_map::Matrix& gridMapObjects = map["objects"];
    grid_map::Matrix& gridMapElevation = map["elevation"];
    
    for(OcTreeT::leaf_bbx_iterator it = m_octree->begin_leafs_bbx(minBbx, maxBbx),
            end = m_octree->end_leafs_bbx(); it != end; ++it) {

        octomap::point3d octoPos = it.getCoordinate();
        grid_map::Position position(octoPos.x(), octoPos.y());
        grid_map::Index index;
        map.getIndex(position, index);

        // if (!m_octree->isNodeOccupied(*it)){
        //     gridMapData(index(0), index(1)) = it->getOccupancy();
        // }

        // int class_label = static_cast<int>(it->getColor().r);
        // std::cout << "Node color " << it->getColor() << std::endl;
        Eigen::Vector3i color_rgb;
        // color_rgb[0] =  static_cast<int>(covins_params::ade20k::class_colors[class_label].mu8R);
        // color_rgb[1] =  static_cast<int>(covins_params::ade20k::class_colors[class_label].mu8G);
        // color_rgb[2] =  static_cast<int>(covins_params::ade20k::class_colors[class_label].mu8B);
        color_rgb[0] =  static_cast<int>(it->getColor().r);
        color_rgb[1] =  static_cast<int>(it->getColor().g);
        color_rgb[2] =  static_cast<int>(it->getColor().b);
        float color_value = 0.0; // default gray
        grid_map::colorVectorToValue(color_rgb, color_value );
        // std::cout << "Color RGB: " << color_rgb << " Color value: " << color_value << std::endl;

        // TODO: cleanup
        bool omit_occupancy = false; // ( (color_rgb[0] == 80 && color_rgb[1] == 50 && color_rgb[2] == 50) ||    // floor 
                                //(color_rgb[0] == 120 && color_rgb[1] == 120 && color_rgb[2] == 80)  ||  // ceiling
                                //(color_rgb[0] == 255 && color_rgb[1] == 173 && color_rgb[2] == 0) ) ;    //lights

        if (!omit_occupancy) {      /// UGH
            if (!map.isValid(index, "occupancy")) {
                gridMapData(index(0), index(1)) = it->getOccupancy();
            }
            else {
                // gridMapData(index(0), index(1)) = gridMapData(index(0), index(1)) +  it->getOccupancy();it->getOccupancy()
                if (gridMapData(index(0), index(1)) < 5 )
                    gridMapData(index(0), index(1)) = gridMapData(index(0), index(1)) + it->getOccupancy();
            } 
            
        
            if (!m_octree->isNodeOccupied(*it)) {
                continue;
            }
            gridMapObjects(index(0), index(1)) = color_value;
            // If no elevation has been set, use current elevation.
            if (!map.isValid(index, "elevation")) {
                gridMapElevation(index(0), index(1)) = octoPos.z();
            }
            else {
            // Check existing elevation, keep higher.
                if (gridMapElevation(index(0), index(1)) < octoPos.z()) {
                    gridMapElevation(index(0), index(1)) = octoPos.z();
                }
            }
        }
        else {ROS_WARN("Skipped a point!");}
    }

    return true;
}

void GridMapper::publishProjected2DMap(const ros::Time &rostime)
{
    // m_gridmap.header.stamp = rostime;

    grid_map::Position3 min_bound;
    grid_map::Position3 max_bound;
    m_octree->getMetricMin(min_bound(0), min_bound(1), min_bound(2));
    m_octree->getMetricMax(max_bound(0), max_bound(1), max_bound(2));
    // if(!std::isnan(minX_))
    //     min_bound(0) = minX_;
    // if(!std::isnan(maxX_))
    //     max_bound(0) = maxX_;
    // if(!std::isnan(minY_))
    //     min_bound(1) = minY_;
    // if(!std::isnan(maxY_))
    //     max_bound(1) = maxY_;
    // if(!std::isnan(minZ_))
    //     min_bound(2) = minZ_;
    // if(!std::isnan(maxZ_))
    //     max_bound(2) = maxZ_;
    // bool res = grid_map::GridMapOctomapConverter::fromOctomap(*m_octree, "occupancy", map, &min_bound, &max_bound);
    bool res = populateGridMap();
    if (!res) {
        ROS_ERROR("Failed to call convert Octomap.");
        return;
    }
    map.setFrameId("world");

    // Publish as grid map.
    grid_map_msgs::GridMap gridMapMessage;
    grid_map::GridMapRosConverter::toMessage(map, gridMapMessage);
    m_gridmapPub.publish(gridMapMessage);

    // Also publish as an octomap msg for visualization
    publishFullOctoMap(rostime);
    // publishMapAsMarkers(rostime, false);
    
}