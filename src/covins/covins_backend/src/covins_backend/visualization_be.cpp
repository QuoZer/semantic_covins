/**
* This file is part of COVINS.
*
* Copyright (C) 2018-2021 Patrik Schmuck / Vision for Robotics Lab
* (ETH Zurich) <collaborative (dot) slam (at) gmail (dot) com>
* For more information see <https://github.com/VIS4ROB-lab/covins>
*
* COVINS is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the license, or
* (at your option) any later version.
*
* COVINS is distributed to support research and development of
* multi-agent system, but WITHOUT ANY WARRANTY; without even the
* implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
* PURPOSE. In no event will the authors be held liable for any damages
* arising from the use of this software. See the GNU General Public
* License for more details.
*
* You should have received a copy of the GNU General Public License
* along with COVINS. If not, see <http://www.gnu.org/licenses/>.
*/

#include "covins_backend/visualization_be.hpp"

// C++
#include <set>
#include <eigen3/Eigen/Core>
#include <limits.h>

// COVINS
#include "covins_backend/landmark_be.hpp"
#include "covins_backend/map_be.hpp"
#include "covins_backend/CloudWallSegmentation.hpp"
#include "covins_backend/post_processor.hpp"

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

// Thirdparty
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

namespace covins {

Visualizer::Visualizer(std::string topic_prefix)
    : VisualizerBase(topic_prefix)
{
    post_processor = new MapPostProcessor(ros::NodeHandle("~"));
    //...
}

auto Visualizer::DrawMap(MapPtr map)->void {
    if(!map) return;
    std::unique_lock<std::mutex> lock(mtx_draw_);
    // std::cout << "Inside DrawMap \n";
    VisBundle vb;
    vb.keyframes = map->GetKeyframes();
    vb.keyframes_erased = map->GetKeyframesErased();
    vb.landmarks = map->GetLandmarks();
    vb.id_map = map->id_map_;
    vb.associated_clients = map->associated_clients_;
    vb.loops = map->GetLoopConstraints();
    vb.frame = "odom";

    if(vb.keyframes.size() < 3) return;

    vis_data_[map->id_map_] = vb;
}

auto Visualizer::PubCovGraph()->void {
    visualization_msgs::Marker cov_graph_msg;
    visualization_msgs::Marker shared_edge_msg;
    std::stringstream ss;

    cov_graph_msg.header.frame_id = curr_bundle_.frame;
    cov_graph_msg.header.stamp = ros::Time::now();
    ss << "CovGraph" << curr_bundle_.id_map << topic_prefix_;
    cov_graph_msg.ns = ss.str();
    cov_graph_msg.type = visualization_msgs::Marker::LINE_LIST;
    covins_params::VisColorRGB col = covins_params::colors::color_cov;
    cov_graph_msg.color = MakeColorMsg(col.mfR,col.mfG,col.mfB);
    cov_graph_msg.action = visualization_msgs::Marker::ADD;
    cov_graph_msg.scale.x = covins_params::vis::covmarkersize;
    cov_graph_msg.id = 0;

    shared_edge_msg = cov_graph_msg;
    ss = std::stringstream();
    ss << "CovGraphSharedEdges" << curr_bundle_.id_map << topic_prefix_;
    shared_edge_msg.ns = ss.str();
    shared_edge_msg.color = MakeColorMsg(1.0,0.0,0.0);

    precision_t scale = covins_params::vis::scalefactor;

    std::set<std::pair<KeyframePtr,KeyframePtr>> covlinks;

    for(KeyframeMap::iterator mit = curr_bundle_.keyframes.begin();mit!=curr_bundle_.keyframes.end();++mit) {
        KeyframePtr kf = mit->second;
        if(kf->IsInvalid()) continue;
//        kf->UpdateCovisibilityConnections(); // one must no call functions that change KFs/LMs/Map here!
        KeyframeVector connected_kfs = kf->GetConnectedKeyframesByWeight(covins_params::vis::covgraph_minweight);

        for(KeyframeVector::iterator vit2 = connected_kfs.begin();vit2!=connected_kfs.end();++vit2) {
            KeyframePtr kf_con = *vit2;
            if(!kf_con || kf_con->IsInvalid()) continue;

            std::pair<KeyframePtr,KeyframePtr> link;
            if(kf->id_ < kf_con->id_)
                link = std::make_pair(kf,kf_con);
            else
                link = std::make_pair(kf_con,kf);

            covlinks.insert(link);
        }
    }

    for(std::set<std::pair<KeyframePtr,KeyframePtr>>::iterator sit = covlinks.begin();sit!=covlinks.end();++sit) {
        KeyframePtr kf0 = sit->first;
        KeyframePtr kf1 = sit->second;

        Eigen::Matrix4d T0 = kf0->GetPoseTws();
        Eigen::Matrix4d T1 = kf1->GetPoseTws();

        geometry_msgs::Point p1;
        geometry_msgs::Point p2;

        p1.x = scale*((double)(T0(0,3)));
        p1.y = scale*((double)(T0(1,3)));
        p1.z = scale*((double)(T0(2,3)));

        p2.x = scale*((double)(T1(0,3)));
        p2.y = scale*((double)(T1(1,3)));
        p2.z = scale*((double)(T1(2,3)));

        if(kf0->id_.second == kf1->id_.second) {
            cov_graph_msg.points.push_back(p1);
            cov_graph_msg.points.push_back(p2);
        } else {
            shared_edge_msg.points.push_back(p1);
            shared_edge_msg.points.push_back(p2);
        }
    }

    if(!covins_params::vis::covgraph_shared_edges_only && !cov_graph_msg.points.empty()) {
        pub_marker_.publish(cov_graph_msg);
    }
    if(!shared_edge_msg.points.empty()) {
        pub_marker_.publish(shared_edge_msg);
    }
}

auto Visualizer::PubKeyframesAsFrusta()->void {
    std::map<int,visualization_msgs::Marker> msgs;
    std::map<int,visualization_msgs::Marker> msgs_erased;

    for(std::set<size_t>::iterator sit = curr_bundle_.associated_clients.begin(); sit != curr_bundle_.associated_clients.end(); ++sit) {
        int cid = *sit;

        visualization_msgs::Marker keyframes;
        keyframes.header.frame_id = curr_bundle_.frame;
        keyframes.header.stamp = ros::Time::now();
        std::stringstream ss;
        ss << "KFs" << cid << topic_prefix_;
        keyframes.ns = ss.str();
        keyframes.id=0;
        keyframes.type = visualization_msgs::Marker::LINE_LIST;
        keyframes.scale.x=covins_params::vis::camlinesize;
        keyframes.pose.orientation.w=1.0;
        keyframes.action=visualization_msgs::Marker::ADD;
        if(cid < 12){
            covins_params::VisColorRGB col = covins_params::colors::col_vec[cid];
            keyframes.color = MakeColorMsg(col.mfR,col.mfG,col.mfB);
        }
        else
            keyframes.color = MakeColorMsg(0.5,0.5,0.5);

        msgs[cid] = keyframes;

        // erased KFs
        ss = std::stringstream();
        ss << "KFs" << cid << "_erased" << topic_prefix_;
        keyframes.ns = ss.str();
        keyframes.color = MakeColorMsg(1.0,0.0,0.0);
        msgs_erased[cid] = keyframes;
    }

    const precision_t scale = covins_params::vis::scalefactor;
    const precision_t d = covins_params::vis::camsize;

    for(KeyframeMap::iterator mit = curr_bundle_.keyframes.begin();mit!=curr_bundle_.keyframes.end();++mit){
        KeyframePtr kf = mit->second;
        size_t cid = kf->id_.second;

        {
            // Camera is a pyramid. Define in camera coordinate system
            Eigen::Vector4d o,p1,p2,p3,p4;
            o << 0, 0, 0, 1;
            p1 << d, d*0.8, d*0.5, 1;
            p2 << d, -d*0.8, d*0.5, 1;
            p3 << -d, -d*0.8, d*0.5, 1;
            p4 << -d, d*0.8, d*0.5, 1;

            if(curr_bundle_.most_recent_kfs.count(kf->id_)){
                // double size for current KFs
                o << 0, 0, 0, 1;
                p1 << MUL*d, MUL*d*0.8, MUL*d*0.5, 1;
                p2 << MUL*d, -MUL*d*0.8, MUL*d*0.5, 1;
                p3 << -MUL*d, -MUL*d*0.8, MUL*d*0.5, 1;
                p4 << -MUL*d, MUL*d*0.8, MUL*d*0.5, 1;
            }

            precision_t factor = 2.0;
            if(kf->id_.first % 10 == 0){
                // double size for current KFs
                o << 0, 0, 0, 1;
                p1 << factor*d, factor*d*0.8, factor*d*0.5, 1;
                p2 << factor*d, -factor*d*0.8, factor*d*0.5, 1;
                p3 << -factor*d, -factor*d*0.8, factor*d*0.5, 1;
                p4 << -factor*d, factor*d*0.8, factor*d*0.5, 1;
            }

            Eigen::Matrix4d Twc = kf->GetPoseTwc();
            Eigen::Vector3d ow = kf->GetPoseTwc().block<3,1>(0,3);
            Eigen::Vector4d p1w = Twc*p1;
            Eigen::Vector4d p2w = Twc*p2;
            Eigen::Vector4d p3w = Twc*p3;
            Eigen::Vector4d p4w = Twc*p4;

            geometry_msgs::Point msgs_o,msgs_p1, msgs_p2, msgs_p3, msgs_p4;
            msgs_o.x=(scale)*ow(0);
            msgs_o.y=(scale)*ow(1);
            msgs_o.z=(scale)*ow(2);
            msgs_p1.x=(scale)*p1w(0);
            msgs_p1.y=(scale)*p1w(1);
            msgs_p1.z=(scale)*p1w(2);
            msgs_p2.x=(scale)*p2w(0);
            msgs_p2.y=(scale)*p2w(1);
            msgs_p2.z=(scale)*p2w(2);
            msgs_p3.x=(scale)*p3w(0);
            msgs_p3.y=(scale)*p3w(1);
            msgs_p3.z=(scale)*p3w(2);
            msgs_p4.x=(scale)*p4w(0);
            msgs_p4.y=(scale)*p4w(1);
            msgs_p4.z=(scale)*p4w(2);

            msgs[cid].points.push_back(msgs_o);
            msgs[cid].points.push_back(msgs_p1);
            msgs[cid].points.push_back(msgs_o);
            msgs[cid].points.push_back(msgs_p2);
            msgs[cid].points.push_back(msgs_o);
            msgs[cid].points.push_back(msgs_p3);
            msgs[cid].points.push_back(msgs_o);
            msgs[cid].points.push_back(msgs_p4);
            msgs[cid].points.push_back(msgs_p1);
            msgs[cid].points.push_back(msgs_p2);
            msgs[cid].points.push_back(msgs_p2);
            msgs[cid].points.push_back(msgs_p3);
            msgs[cid].points.push_back(msgs_p3);
            msgs[cid].points.push_back(msgs_p4);
            msgs[cid].points.push_back(msgs_p4);
            msgs[cid].points.push_back(msgs_p1);
        }
    }

    for(KeyframeMap::iterator mit = curr_bundle_.keyframes_erased.begin();mit!=curr_bundle_.keyframes_erased.end();++mit){
        KeyframePtr kf = mit->second;
        size_t cid = kf->id_.second;

        // Camera is a pyramid. Define in camera coordinate system
        Eigen::Vector4d o,p1,p2,p3,p4;
        o << 0, 0, 0, 1;
        p1 << d, d*0.8, d*0.5, 1;
        p2 << d, -d*0.8, d*0.5, 1;
        p3 << -d, -d*0.8, d*0.5, 1;
        p4 << -d, d*0.8, d*0.5, 1;

        Eigen::Matrix4d Twc = kf->GetPoseTwc();
        Eigen::Vector3d ow = kf->GetPoseTwc().block<3,1>(0,3);
        Eigen::Vector4d p1w = Twc*p1;
        Eigen::Vector4d p2w = Twc*p2;
        Eigen::Vector4d p3w = Twc*p3;
        Eigen::Vector4d p4w = Twc*p4;

        geometry_msgs::Point msgs_o,msgs_p1, msgs_p2, msgs_p3, msgs_p4;
        msgs_o.x=(scale)*ow(0);
        msgs_o.y=(scale)*ow(1);
        msgs_o.z=(scale)*ow(2);
        msgs_p1.x=(scale)*p1w(0);
        msgs_p1.y=(scale)*p1w(1);
        msgs_p1.z=(scale)*p1w(2);
        msgs_p2.x=(scale)*p2w(0);
        msgs_p2.y=(scale)*p2w(1);
        msgs_p2.z=(scale)*p2w(2);
        msgs_p3.x=(scale)*p3w(0);
        msgs_p3.y=(scale)*p3w(1);
        msgs_p3.z=(scale)*p3w(2);
        msgs_p4.x=(scale)*p4w(0);
        msgs_p4.y=(scale)*p4w(1);
        msgs_p4.z=(scale)*p4w(2);

        msgs_erased[cid].points.push_back(msgs_o);
        msgs_erased[cid].points.push_back(msgs_p1);
        msgs_erased[cid].points.push_back(msgs_o);
        msgs_erased[cid].points.push_back(msgs_p2);
        msgs_erased[cid].points.push_back(msgs_o);
        msgs_erased[cid].points.push_back(msgs_p3);
        msgs_erased[cid].points.push_back(msgs_o);
        msgs_erased[cid].points.push_back(msgs_p4);
        msgs_erased[cid].points.push_back(msgs_p1);
        msgs_erased[cid].points.push_back(msgs_p2);
        msgs_erased[cid].points.push_back(msgs_p2);
        msgs_erased[cid].points.push_back(msgs_p3);
        msgs_erased[cid].points.push_back(msgs_p3);
        msgs_erased[cid].points.push_back(msgs_p4);
        msgs_erased[cid].points.push_back(msgs_p4);
        msgs_erased[cid].points.push_back(msgs_p1);
    }

    for(std::map<int,visualization_msgs::Marker>::iterator mit=msgs.begin();mit!=msgs.end();++mit){
        visualization_msgs::Marker msg = mit->second;
        if(!msg.points.empty()){
            pub_marker_.publish(msg);
        }
    }

    for(std::map<int,visualization_msgs::Marker>::iterator mit=msgs_erased.begin();mit!=msgs_erased.end();++mit){
        visualization_msgs::Marker msg = mit->second;
        if(!msg.points.empty()){
            pub_marker_.publish(msg);
        }
    }
}

auto Visualizer::PubLandmarksAsCloud()->void {
    pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
    pcl::PointCloud<pcl::PointXYZRGBNormal> cloud_filtered;
    pcl::PointCloud<pcl::PointXYZL> cloud_labeled;

    cloud.width = 0;
    cloud.height = 1;
    cloud_filtered.width = 0;
    cloud_filtered.height = 1;
    cloud_labeled.width = 0;
    cloud_labeled.height = 1;

    // CloudWallSegmentation cloudSegmentation;
    // std::vector<int> certainty(150); 
    // std::vector<int> pt_count(150); 

    for(LandmarkMap::const_iterator mit=curr_bundle_.landmarks.begin();mit!=curr_bundle_.landmarks.end();++mit) {
        LandmarkPtr lm_i = mit->second;
        Eigen::Vector3d PosWorld = lm_i->GetWorldPos();
        
        int class_label = AggregateLabels(lm_i);

        // std::pair<int,int> cls_diff = AggregateLabelsExt(lm_i);
        // int class_label = cls_diff.first;
        // certainty[class_label] += cls_diff.second; 
        // pt_count[class_label] += 1; 

        // p = CreatePoint3D(PosWorld,lm_i->id_.second);
        pcl::PointXYZRGB p;
        p = CreateSemanticPoint3D(PosWorld,lm_i->id_.second, class_label);
        // get eigen normal vector
        TypeDefs::Vector3Type lm_normal = lm_i->GetNormal();
        pcl::PointXYZRGBNormal np;
        np.x = p.x;
        np.y = p.y;
        np.z = p.z;
        np.r = 255-p.r;         // yeah, idk, colors don't look right the other way 
        np.g = 255-p.g;
        np.b = 255-p.b;
        // np.normal = lm_normal;
        np.normal_x = lm_normal[0];
        np.normal_y = lm_normal[1];
        np.normal_z = lm_normal[2];
        pcl::PointXYZL lp;
        lp.x = p.x;
        lp.y = p.y;
        lp.z = p.z;
        lp.label = class_label;

        cloud.points.push_back(np);
        cloud.width++;
        // removing the floor and ceiling lms
        if (class_label == 5 || class_label == 3)
            continue; 
        cloud_filtered.points.push_back(np);
        cloud_filtered.width++;

        cloud_labeled.points.push_back(lp);
        cloud_labeled.width++;
    }
    
    // for (int c = 0; c < 150; c++)
    // {
    //     if (pt_count[c] == 0) continue;
    //     std::cout << "For class " << c+1 << " the average confidence is " << certainty[c] / pt_count[c] << "%\n"; 
    // }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    KeyframeSetById kftraj;
    std::set<int> kfcids;

    for(KeyframeMap::iterator mit = curr_bundle_.keyframes.begin();mit!=curr_bundle_.keyframes.end();++mit){
        KeyframePtr kf = mit->second;
        kftraj.insert(kf);
        kfcids.insert(kf->id_.second);
    }

    // int timeStamp = 0;
    covins_backend::trajectories_message trajectoriesMsg;
    trajectoriesMsg.agent_number=0;
    trajectoriesMsg.sizes.clear();
    trajectoriesMsg.timestamps.clear();
    trajectoriesMsg.data.clear();
    

    std::map<int,visualization_msgs::Marker> msgs;
    for(std::set<int>::iterator sit = kfcids.begin(); sit != kfcids.end(); ++sit){
        int cid = *sit;
        visualization_msgs::Marker traj;
        traj.header.frame_id = curr_bundle_.frame;
        traj.header.stamp = ros::Time::now();
        // if(timeStamp == 0) timeStamp = (int)traj.header.stamp.toSec();
        int timeS = (int)traj.header.stamp.toSec();
        //trajectoriesMsg.timestamps.push_back(timeS);
        std::stringstream ss;
        ss << "Traj" << cid << topic_prefix_;
        traj.ns = ss.str();
        traj.id=0;
        traj.type = visualization_msgs::Marker::LINE_STRIP;
        traj.scale.x=covins_params::vis::trajmarkersize;
        traj.action=visualization_msgs::Marker::ADD;

        if(cid < 12){
            covins_params::VisColorRGB col = covins_params::colors::col_vec[cid];
            traj.color = MakeColorMsg(col.mfR,col.mfG,col.mfB);
        }
        else
            traj.color = MakeColorMsg(0.5,0.5,0.5);

        msgs[cid] = traj;
    }

    precision_t scale = covins_params::vis::scalefactor;
    std::vector<float> trajectory;

    size_t lastID = 100;
    for(KeyframeSetById::iterator sit = kftraj.begin(); sit != kftraj.end(); ++sit){
        KeyframePtr kf = *sit;
        if (lastID != kf->id_.second){
            lastID = kf->id_.second;
            if (!trajectory.empty()) {
                trajectoriesMsg.agent_number++;
                trajectoriesMsg.sizes.push_back(trajectory.size());
                trajectory.clear();
            }
        }
        Eigen::Matrix4d T = kf->GetPoseTws();
        geometry_msgs::Point p;
        p.x = scale*(T(0,3));
        p.y = scale*(T(1,3));
        p.z = scale*(T(2,3));
        float fx = (float)(scale*(T(0,3)));
        float fy = (float)(scale*(T(1,3)));
        trajectory.push_back(fx);
        trajectory.push_back(fy);
        trajectoriesMsg.data.push_back(fx);
        trajectoriesMsg.data.push_back(fy);
        MsgKeyframe msg_kf;
        //kf->ConvertToMsg(msg_kf,kf,false);
        //trajectoriesMsg.timestamps.push_back(msg_kf.timestamp);
        trajectoriesMsg.timestamps.push_back(kf->timestamp_);
        msgs[kf->id_.second].points.push_back(p);
    }
    if (!trajectory.empty()) {
        trajectoriesMsg.agent_number++;
        trajectoriesMsg.sizes.push_back(trajectory.size());
        trajectory.clear();
    }
   
    for(std::map<int,visualization_msgs::Marker>::iterator mit = msgs.begin(); mit != msgs.end(); ++mit){
        visualization_msgs::Marker msg = mit->second;
        pub_marker_.publish(msg);
    }

/////////////////////////////////////////////////////////////////////////////////////////////////////

    if(!cloud.points.empty())
    {
        sensor_msgs::PointCloud2 pcl_msg;
        sensor_msgs::PointCloud2 pcl_filtered_msg;
        sensor_msgs::PointCloud2 pcl_labeled_msg;
        pcl::toROSMsg(cloud,pcl_msg);
        pcl::toROSMsg(cloud_filtered, pcl_filtered_msg);
        pcl::toROSMsg(cloud_labeled, pcl_labeled_msg);
        // COMMENTING OUT CLOUD FILTERS
        // outputCloud = cloudSegmentation.processPointcloud(cloud);
        // pcl::toROSMsg(outputCloud,pcl_filtered_msg);
        // pcl_filtered_msg = pcl_msg;
        // cv::Mat projection = cloudSegmentation.createProjectionImage(cloud, 2400, 1080);
        // cv::Mat projection_trajectory = cloudSegmentation.DrawTrajectoriesTime2(trajectoriesMsg.data, trajectoriesMsg.sizes, projection.clone(), trajectoriesMsg.timestamps, 20000000);
        // cv::imwrite("/var/www/pointmap.net/html/imgs/bw_walls1.png",projection_trajectory);
        // system("rm /var/www/pointmap.net/html/imgs/bw_walls.png");
        // system("mv /var/www/pointmap.net/html/imgs/bw_walls1.png /var/www/pointmap.net/html/imgs/bw_walls.png");
        
        pcl_msg.header.frame_id = curr_bundle_.frame;
        pcl_msg.header.stamp = ros::Time::now();
        pcl_filtered_msg.header.frame_id = curr_bundle_.frame;
        pcl_filtered_msg.header.stamp = ros::Time::now();
        pcl_labeled_msg.header.frame_id = curr_bundle_.frame;
        pcl_labeled_msg.header.stamp = ros::Time::now();

        pub_cloud_.publish(pcl_msg);
        pub_cloud_filtered_.publish(pcl_filtered_msg);
        pub_cloud_labeled_.publish(pcl_labeled_msg);
        pub_trajectories_.publish(trajectoriesMsg);
    }
}

auto Visualizer::PubLastLandmarksAsCloud()->void {
    KeyframePtr kf = curr_bundle_.keyframes.crbegin()->second;
    std::list<int> vec = {};
    PubKFAsCloud(kf, vec);
}

auto Visualizer::PubKFAsCloud(KeyframePtr kf, std::list<int> &filtered_lms)->void {
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cloud.width = 0;
    cloud.height = 1;
        
    covins::TypeDefs::TransformType camera_origin = kf->GetPoseTws();

    // transform world->camera
    geometry_msgs::TransformStamped camera_tf;
    camera_tf.header.stamp = ros::Time::now();
    camera_tf.header.frame_id = "odom";
    camera_tf.child_frame_id = "camera" + std::to_string(kf->id_.second);
    camera_tf.transform.translation.x = camera_origin(0,3);
    camera_tf.transform.translation.y = camera_origin(1,3);
    camera_tf.transform.translation.z = camera_origin(2,3);
    Eigen::Quaterniond q(camera_origin.block<3,3>(0,0));
    camera_tf.transform.rotation.x = q.x();
    camera_tf.transform.rotation.y = q.y();
    camera_tf.transform.rotation.z = q.z();
    camera_tf.transform.rotation.w = q.w();
    tf_broadcaster.sendTransform(camera_tf);

    // walls, doors, curtains, painting, poster, | floor, ceiling, lights
    std::vector<int> obstacle_classes = {0, 14, 18, 22, 100,3, 5, 82 }; // TODO: get from the file 

    covins::TypeDefs::LandmarkVector kf_landmarks = kf->GetValidLandmarks();
    for (covins::TypeDefs::LandmarkPtr lm : kf_landmarks)
    {       
        if ( filtered_lms.size() > 1 && std::find(filtered_lms.begin(), filtered_lms.end(), lm->id_.second) == filtered_lms.end() )
            continue;       // publish only filtered points 
        int class_label = AggregateLabels(lm);

        // check is the label is in obstacle classes, otherwise skip 
        if (std::find(obstacle_classes.begin(), obstacle_classes.end(), class_label) == obstacle_classes.end() )
            continue;
        
        pcl::PointXYZRGB p;

        Eigen::Vector3d PosWorld = lm->GetWorldPos();
        Eigen::Vector3d PosCamera = camera_origin.block<3,3>(0,0).transpose()*(PosWorld - camera_origin.block<3,1>(0,3));
        
        // p = CreatePoint3D(PosWorld,lm_i->id_.second);
        p = CreateSemanticPoint3D(PosCamera,lm->id_.second, class_label);
        // get eigen normal vector
        TypeDefs::Vector3Type lm_normal = lm->GetNormal();
        pcl::PointXYZRGB np;
        np.x = p.x;
        np.y = p.y;
        np.z = p.z;
        np.r = 255-p.r;
        np.g = 255-p.g;
        np.b = 255-p.b;

        cloud.points.push_back(p);
        cloud.width++;
    }

    // Create the filtering object
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
    // pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    // sor.setInputCloud(cloud.makeShared());
    // sor.setMeanK(35);
    // sor.setStddevMulThresh(1.0);
    // // sor.setNegative (true);
    // sor.filter(*cloud_filtered);    

    if(!cloud.points.empty())
    {
        sensor_msgs::PointCloud2 pcl_msg;
        pcl::toROSMsg(cloud,pcl_msg);
        pcl_msg.header.stamp = ros::Time::now();
        pcl_msg.header.frame_id = "camera" + std::to_string(kf->id_.second);

        // if (kf->id_.second == 0)
        //     pub_keyframes_1_.publish(pcl_msg);
        // else if (kf->id_.second == 1)
        //     pub_keyframes_2_.publish(pcl_msg);
        // pub_cloud_.publish(pcl_msg);

        pub_keyframes_2_.publish(pcl_msg);
    }
}

auto Visualizer::AggregateLabels(LandmarkPtr lm_p)->int {
    std::map<int, int> class_dest;  // [class|n_observations] destribution of observation classes
    
    TypeDefs::KfObservations kf_obs = lm_p->GetObservations();
    for (TypeDefs::KfObservations::iterator observation = kf_obs.begin(); observation != kf_obs.end(); ++observation) {
        KeyframePtr kf_candidate = observation->first;
        if (kf_candidate->keypoint_classes.empty())
            continue;
        int obs_idx = kf_candidate->GetLandmarkIndex(lm_p); // same as lm->second, lol
        int class_candidate = kf_candidate->keypoint_classes[obs_idx];
        // Increment the class' probability
        class_dest[class_candidate]++;
    }
    // Finding max value
    std::map<int,int>::iterator max_class_count
        = std::max_element(class_dest.begin(),class_dest.end(),
        [] (const std::pair<int,int>& a, const std::pair<int,int>& b)->bool{ return a.second < b.second; } );
    // Finding the matching key 
    int class_key = max_class_count->first;

    return class_key; 
}

// a slight modification of the function above that outputs the different with the next prediction in addition to the semantic class 
auto Visualizer::AggregateLabelsExt(LandmarkPtr lm_p)->std::pair<int,int> {
    std::map<int, int> class_dest;  // [class|n_observations] destribution of observation classes
    
    TypeDefs::KfObservations kf_obs = lm_p->GetObservations();
    for (TypeDefs::KfObservations::iterator observation = kf_obs.begin(); observation != kf_obs.end(); ++observation) {
        KeyframePtr kf_candidate = observation->first;
        if (kf_candidate->keypoint_classes.empty())
            continue;
        int obs_idx = kf_candidate->GetLandmarkIndex(lm_p); // same as lm->second, lol
        int class_candidate = kf_candidate->keypoint_classes[obs_idx];
        // Increment the class' probability
        class_dest[class_candidate]++;
    }
    // Finding max value
    std::map<int,int>::iterator max_class_count
        = std::max_element(class_dest.begin(),class_dest.end(),
        [] (const std::pair<int,int>& a, const std::pair<int,int>& b)->bool{ return a.second < b.second; } );
    // Finding the matching key 
    int class_key = max_class_count->first;

    // Find the second max value 
    // Remove the max element from the map
    class_dest.erase(max_class_count);

    // Find the second max value
    std::map<int,int>::iterator second_max_class_count
        = std::max_element(class_dest.begin(),class_dest.end(),
        [] (const std::pair<int,int>& a, const std::pair<int,int>& b)->bool{ return a.second < b.second; } );

    // Finding the matching key 
    int second_class_key = second_max_class_count->first;

    // int diff = max_class_count->second - second_max_class_count->second; 
    int diff = 100 * max_class_count->second / kf_obs.size();

    return std::pair<int,int>{class_key, diff}; 
}

auto Visualizer::PubTrajectories()->void {
    KeyframeSetById kftraj;
    std::set<int> kfcids;

    for(KeyframeMap::iterator mit = curr_bundle_.keyframes.begin();mit!=curr_bundle_.keyframes.end();++mit){
        KeyframePtr kf = mit->second;
        kftraj.insert(kf);
        kfcids.insert(kf->id_.second);
    }

    std::map<int,visualization_msgs::Marker> msgs;
    for(std::set<int>::iterator sit = kfcids.begin(); sit != kfcids.end(); ++sit){
        int cid = *sit;
        visualization_msgs::Marker traj;
        traj.header.frame_id = curr_bundle_.frame;
        traj.header.stamp = ros::Time::now();
        std::stringstream ss;
        ss << "Traj" << cid << topic_prefix_;
        traj.ns = ss.str();
        traj.id=0;
        traj.type = visualization_msgs::Marker::LINE_STRIP;
        traj.scale.x=covins_params::vis::trajmarkersize;
        traj.action=visualization_msgs::Marker::ADD;

        if(cid < 12){
            covins_params::VisColorRGB col = covins_params::colors::col_vec[cid];
            traj.color = MakeColorMsg(col.mfR,col.mfG,col.mfB);
        }
        else
            traj.color = MakeColorMsg(0.5,0.5,0.5);

        msgs[cid] = traj;
    }

    precision_t scale = covins_params::vis::scalefactor;

    for(KeyframeSetById::iterator sit = kftraj.begin(); sit != kftraj.end(); ++sit){
        KeyframePtr kf = *sit;
        Eigen::Matrix4d T = kf->GetPoseTws();
        geometry_msgs::Point p;
        p.x = scale*(T(0,3));
        p.y = scale*(T(1,3));
        p.z = scale*(T(2,3));
        msgs[kf->id_.second].points.push_back(p);
    }

    for(std::map<int,visualization_msgs::Marker>::iterator mit = msgs.begin(); mit != msgs.end(); ++mit){
        visualization_msgs::Marker msg = mit->second;

        //std::cout<<msg;

        pub_marker_.publish(msg);
    }
}

void Visualizer::RepubMap(std::list<int> &lm_ids_include, int sleep_us)
{
    std_msgs::Bool done;
    done.data = false;
    // int kf_count = 0;
    for(KeyframeMap::iterator mit = curr_bundle_.keyframes.begin();mit!=curr_bundle_.keyframes.end();++mit){
        KeyframePtr kf = mit->second;

        PubKFAsCloud(kf, lm_ids_include);
        
        // grid mapping related code
        // if (kf_count++ % 2 == 0)
        //     oc_grid->insertKF(kf);

        usleep(sleep_us);
        post_processor->m_statePub.publish(done);
    }  
    PubLandmarksAsCloud();
    usleep(1000000);
    done.data = true;
    post_processor->m_statePub.publish(done);
}

void Visualizer::UpdateAndPubGrid(int type)
{
    post_processor->updateTransform();

    pcl::PointCloud<pcl::PointXYZL>::Ptr full_cloud = post_processor->createLabeledCloud(curr_bundle_.landmarks);
    // whitelisted landmark ids  TODO: switch to blacklist 
    std::list<int> lm_ids;
    for (auto lm : curr_bundle_.landmarks) {
        // lm_ids[i] = lm.second->id_.second; 
        lm_ids.push_back(lm.second->id_.second);
    }
    std::cout << "Starting with lm_ids of size: " << lm_ids.size() << " and ptc of size: " << full_cloud->size() << "\n";

    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZL>);
    if (type > 0) 
    {
        // Create the filtering object
        pcl::StatisticalOutlierRemoval<pcl::PointXYZL> sor(true);
        sor.setInputCloud(full_cloud);
        sor.setMeanK(10);
        sor.setStddevMulThresh(0.00005);
        sor.filter(*cloud_filtered);   
        pcl::PointIndices::Ptr rmd_indices(new pcl::PointIndices);
        sor.getRemovedIndices(*rmd_indices); 

        // save for debug
        pcl::io::savePCDFileASCII("/home/appuser/COVINS_demo/stat_cloud.pcd", *cloud_filtered);  
        std::cout << "Statistical filter removed "  << rmd_indices->indices.size() << " indices, left with: " << cloud_filtered->size() <<"\n";

        int erased_lms = 0;
        for (int index : rmd_indices->indices) {
            std::list<int>::iterator it = lm_ids.begin();
            std::advance(it, index - erased_lms);
            lm_ids.erase(it);
            erased_lms++;
        }
    }
    else {
        cloud_filtered = full_cloud;
    }

    if ( type > 1)
    {
        std::vector<int> ignored_classes = {3, 5, 82};
        std::vector<int> remaining_classes(covins_params::ade20k::obstacle_ids.size() + covins_params::ade20k::floor_ceil_ids.size());
        std::merge(covins_params::ade20k::obstacle_ids.begin(), covins_params::ade20k::obstacle_ids.end(), 
                covins_params::ade20k::floor_ceil_ids.begin(), covins_params::ade20k::floor_ceil_ids.end(),
                remaining_classes.begin());
        pcl::PointIndices::Ptr noise_indices = post_processor->removeFurnitureNoise(cloud_filtered, lm_ids, remaining_classes, ignored_classes, 0.5, 8, 100);
    }
    std::cout << "Finishing with lm_ids of size: " << lm_ids.size() << "\n";

    PubLandmarksAsCloud();

    RepubMap(lm_ids, 7500);
   
    std::cout << "Map published \n";
}

auto Visualizer::PubLoopEdges()->void {
    if(curr_bundle_.keyframes.empty()){
        std::cout << COUTWARN << "no KFs on VisBundle" << std::endl;
        return;
    }

    LoopVector loops = curr_bundle_.loops;

    if(loops.empty())
        return;

    visualization_msgs::Marker msg_intra; // loops between KFs from the same agent
    visualization_msgs::Marker msg_inter; // loops between KFs from different agents

    msg_intra.header.frame_id = curr_bundle_.frame;
    msg_intra.header.stamp = ros::Time::now();
    std::stringstream ss;
    ss << "LoopEdges_intra_" << curr_bundle_.id_map << topic_prefix_ << std::endl;
    msg_intra.ns = ss.str();
    msg_intra.type = visualization_msgs::Marker::LINE_LIST;
    msg_intra.color = MakeColorMsg(1.0,0.0,0.0);
    msg_intra.action = visualization_msgs::Marker::ADD;
    msg_intra.scale.x = covins_params::vis::loopmarkersize;
    msg_intra.id = 0;

    msg_inter.header.frame_id = curr_bundle_.frame;
    msg_inter.header.stamp = ros::Time::now();
    ss = std::stringstream();
    ss << "LoopEdges_inter_" << curr_bundle_.id_map << topic_prefix_ << std::endl;
    msg_inter.ns = ss.str();
    msg_inter.type = visualization_msgs::Marker::LINE_LIST;
    msg_inter.color = MakeColorMsg(1.0,0.0,0.0);
    msg_inter.action = visualization_msgs::Marker::ADD;
    msg_inter.scale.x = covins_params::vis::loopmarkersize;
    msg_inter.id = 1;

    precision_t scale = covins_params::vis::scalefactor;

    for(size_t idx = 0; idx<loops.size(); ++idx) {
        KeyframePtr kf = loops[idx].kf1;
        KeyframePtr kf2 = loops[idx].kf2;
        bool b_intra = (kf->id_.second == kf2->id_.second);

        Eigen::Matrix4d T1 = kf->GetPoseTws();
        Eigen::Matrix4d T2 = kf2->GetPoseTws();

        geometry_msgs::Point p1;
        geometry_msgs::Point p2;

        p1.x = scale*((double)(T1(0,3)));
        p1.y = scale*((double)(T1(1,3)));
        p1.z = scale*((double)(T1(2,3)));

        p2.x = scale*((double)(T2(0,3)));
        p2.y = scale*((double)(T2(1,3)));
        p2.z = scale*((double)(T2(2,3)));

        if(b_intra){
            msg_intra.points.push_back(p1);
            msg_intra.points.push_back(p2);
        } else {
            msg_inter.points.push_back(p1);
            msg_inter.points.push_back(p2);
        }
    }

    pub_marker_.publish(msg_intra);
    pub_marker_.publish(msg_inter);
}

auto Visualizer::Run()->void {
    while(1) {
        if(this->CheckVisData()) {
            std::unique_lock<std::mutex> lock(mtx_draw_);
            for(std::map<size_t,VisBundle>::iterator mit = vis_data_.begin();mit!=vis_data_.end();++mit){
                curr_bundle_ = mit->second;

                if(covins_params::vis::showkeyframes)
                   this->PubKeyframesAsFrusta();
         
                if(covins_params::vis::showtraj)
                   this->PubTrajectories();

                if(covins_params::vis::showlandmarks) {
                    this->PubLandmarksAsCloud();
                    // this->PubLastLandmarksAsCloud();
                }

                if(covins_params::vis::showcovgraph)
                   this->PubCovGraph();

                this->PubLoopEdges();
            }
            // bool showgrid = true;
            // if (showgrid)
            // if (vis_data_.size() == 1 && showgrid)
            //     this->UpdateAndPubGrid();

            vis_data_.clear();
        }
        this->ResetIfRequested();
        usleep(5000);
    }
}

} //end ns
