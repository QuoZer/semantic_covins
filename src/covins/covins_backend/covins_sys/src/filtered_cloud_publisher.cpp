/**
* This file is not part part of COVINS.
*
* This was made by me: Sanya;)
*/

// COVINS
#include "covins_backend/backend.hpp"
#include "covins_backend/CloudWallSegmentation.hpp"
#include "covins_backend/trajectories_message.h"

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

// C++
#include <iostream>
#include <csignal>
#include <cstdlib>
#include <fstream>


class filtered_cloud_publisher
{
    private:
    
    // System
    ros::NodeHandle nh;
    
    // Messages and Services
    ros::Subscriber sub_filteredCloud;
    ros::Subscriber sub_trajectories;
    ros::Subscriber sub_cloud;
   
    // Parameters
    std::string topic_filteredCloud;
    std::string topic_trajectories;
    std::string topic_cloud;
    
    covins_backend::trajectories_message trajectoriesMsg;
    std::vector<std::vector<float>> trajectories;
    vector<int> timeStamps;
    bool trajReceived;

    public:
    filtered_cloud_publisher();
    ~filtered_cloud_publisher();
    void Callback_InputFilteredCloud(const sensor_msgs::PointCloud2 msg);
    void Callback_InputCloud(const sensor_msgs::PointCloud2 msg);
    void Callback_InputTrajectories(const covins_backend::trajectories_message msg);

    // Segment
    CloudWallSegmentation cloudSegmentation;
};
    
filtered_cloud_publisher::filtered_cloud_publisher():nh("~")
{
    // Parameters
    topic_filteredCloud = "/covins_cloud_filtered";
    topic_trajectories = "/covins_trajectories";
    topic_cloud = "/covins_cloud_be";
    trajReceived=false;
    trajectoriesMsg=covins_backend::trajectories_message();

    // Subscribers and Publishers
    sub_filteredCloud = nh.subscribe<sensor_msgs::PointCloud2>(topic_filteredCloud, 10, &filtered_cloud_publisher::Callback_InputFilteredCloud, this);
    sub_cloud = nh.subscribe<sensor_msgs::PointCloud2>(topic_cloud, 10, &filtered_cloud_publisher::Callback_InputCloud, this);
    sub_trajectories = nh.subscribe<covins_backend::trajectories_message>(topic_trajectories, 10, &filtered_cloud_publisher::Callback_InputTrajectories, this);
}

filtered_cloud_publisher::~filtered_cloud_publisher()
{
    // nothing yet
}

void filtered_cloud_publisher::Callback_InputTrajectories(const covins_backend::trajectories_message msg)
{
    trajectoriesMsg=msg;
    if(!trajReceived){
        trajReceived=true;
    }
}

void filtered_cloud_publisher::Callback_InputCloud(const sensor_msgs::PointCloud2 msg)
{
    if(trajReceived){
        pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud, filtered_cloud;
        
        pcl::fromROSMsg(msg,pcl_cloud);
        filtered_cloud = cloudSegmentation.processPointcloud(pcl_cloud);
        cv::Mat projection = cloudSegmentation.createProjectionImage(filtered_cloud, 2400, 1080);
        cv::Mat projection_trajectory = cloudSegmentation.DrawTrajectoriesTime2(trajectoriesMsg.data,trajectoriesMsg.sizes, projection.clone(), trajectoriesMsg.timestamps, 20000000);
        cv::imwrite("/var/www/pointmap.net/html/imgs/bw_walls1.png",projection_trajectory);
        system("rm /var/www/pointmap.net/html/imgs/bw_walls.png");
        system("mv /var/www/pointmap.net/html/imgs/bw_walls1.png /var/www/pointmap.net/html/imgs/bw_walls.png");
        trajReceived=false;
    }  
}

void filtered_cloud_publisher::Callback_InputFilteredCloud(const sensor_msgs::PointCloud2 msg)
{
    // if(trajReceived){
    //     pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
    //     pcl::fromROSMsg(msg,pcl_cloud);
    //     CloudWallSegmentation cloudSegmentation;
    //     cv::Mat projection = cloudSegmentation.createProjectionImage(pcl_cloud, 2400, 1080);
    //     cv::Mat projection_trajectory = cloudSegmentation.DrawTrajectoriesTime2(trajectoriesMsg.data,trajectoriesMsg.sizes, projection.clone(), trajectoriesMsg.timestamps, 20000000);
    //     cv::imwrite("/var/www/pointmap.net/html/imgs/bw_walls1.png",projection_trajectory);
    //     system("rm /var/www/pointmap.net/html/imgs/bw_walls.png");
    //     system("mv /var/www/pointmap.net/html/imgs/bw_walls1.png /var/www/pointmap.net/html/imgs/bw_walls.png");
    // }  
}

int main(int argc, char **argv)
{
    ros::init(argc,argv,"devtools_path_node");

    filtered_cloud_publisher filtered_cloud_publisher_object;

    ros::spin();

    return 0;
}