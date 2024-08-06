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

#pragma once

// COVINS
#include "covins_base/visualization_base.hpp"
#include "covins_backend/keyframe_be.hpp"
#include <pcl/filters/statistical_outlier_removal.h>

#include "covins_backend/post_processor.hpp"

// Thirdparty
#include <ros/ros.h>

// forward declaration
class MapPostProcessor;
namespace covins {


class Visualizer : public VisualizerBase, public std::enable_shared_from_this<Visualizer> {
public:
    using KeyframeSetById               = std::set<KeyframePtr,Keyframe::kf_less,Eigen::aligned_allocator<KeyframePtr>>;

public:
    Visualizer(std::string topic_prefix = std::string());

    // Main
    virtual auto Run()                                                                  ->void override;

    // Interfaces
    virtual auto DrawMap(MapPtr map)                                                    ->void;

    // Draw Loaded Map
    auto DrawMapBitByBit(MapPtr map, std::string frame)                                 ->void;

    // Resolve final semantic label for a landmark
    static auto AggregateLabels(LandmarkPtr lm_p)                                       ->int;
    static auto AggregateLabelsExt(LandmarkPtr lm_p)                                       ->std::pair<int,int>;

    MapPostProcessor* post_processor; 

    /* 
        Itearte over all keyframes, build a map and publish
        @todo Make it really incremental, check performance 
        @param type level of map filtering before punlishing:
              0 - no filters, raw pointcloud
              1 - statistical outlier removal
              2 - statistical + semantic 
    */
    void UpdateAndPubGrid(int type);

    /*
        Iterates over all keyframes in the map and publishes their respective landmarks if they are in the lm_ids_include list. Sleeps for the set time between messages. Emulates a lidar / depth camera. 
    */
    void RepubMap(std::list<int> &lm_ids_include, int sleep_us=50000);
    
    // tf broadcaster for keyframe position
    tf::TransformBroadcaster tf_broadcaster;

protected:
    // Draw Map
    virtual auto PubCovGraph()                                                          ->void;
    virtual auto PubKeyframesAsFrusta()                                                 ->void;
    virtual auto PubLandmarksAsCloud()                                                  ->void;
    virtual auto PubLastLandmarksAsCloud()                                              ->void;
    virtual auto PubKFAsCloud(KeyframePtr kf, std::list<int> &filtered_lms)                                           ->void;
    virtual auto PubLoopEdges()                                                         ->void;
    virtual auto PubTrajectories()                                                      ->void;
};

} //end ns
