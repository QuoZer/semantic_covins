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

//C++
#include <iostream>
#include <opencv2/opencv.hpp>

//COVINS
#include "typedefs_base.hpp"

namespace covins_params {

using precision_t = covins::TypeDefs::precision_t;

const std::string s0_comm (__FILE__);
const std::size_t p0_comm = s0_comm.find("include/covins");
const std::string s1_comm (s0_comm.substr(0,p0_comm));
const std::string s2_comm ("config/config_comm.yaml");
const std::string s3_comm = s1_comm + s2_comm;
const std::string conf_comm (s3_comm);


// the same function as in config_backend bc I was tired of fixing the linking problems
inline std::vector<int> LoadCategoryIds(std::string path, std::string key)
{
    std::vector<int> ids;
    if (estd2::GetValFromYaml<int>(path,"vis.showsemantics") == 0)
        return ids;     // semantics disabled 

    std::string filename = estd2::GetStringFromYaml(path,"vis.dataset_config_path");
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open dataset description " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::FileNode idsNode = fs[key];

    if (idsNode.type() != cv::FileNode::SEQ) {
        std::cerr << "Invalid format in " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    for (const auto& idNode : idsNode) {
        int id;
        idNode >> id;
        ids.push_back(id);
    }

    return ids;    
}

namespace sys {
    const std::string server_ip                         = estd2::GetStringFromYaml(conf_comm,"sys.server_ip");
    const std::string port                              = estd2::GetStringFromYaml(conf_comm,"sys.port");
}

namespace comm {
    const bool send_updates                             = estd2::GetValFromYaml<bool>(conf_comm,"comm.send_updates");
    const bool data_to_client                           = estd2::GetValFromYaml<bool>(conf_comm,"comm.data_to_client");
    const int start_sending_after_kf                    = estd2::GetValFromYaml<int>(conf_comm,"comm.start_sending_after_kf");
    const int kf_buffer_withold                         = estd2::GetValFromYaml<int>(conf_comm,"comm.kf_buffer_withold");
    const int max_sent_kfs_per_iteration                = estd2::GetValFromYaml<int>(conf_comm,"comm.max_sent_kfs_per_iteration");
    const int update_window_size                        = estd2::GetValFromYaml<int>(conf_comm,"comm.update_window_size");
    const precision_t to_agent_freq                     = estd2::GetValFromYaml<precision_t>(conf_comm,"comm.to_agent_freq");
}

namespace orb {
    const bool activate_visualization                   = estd2::GetValFromYaml<bool>(conf_comm,"orb.activate_visualization");
    const precision_t imu_stamp_max_diff                = estd2::GetValFromYaml<precision_t>(conf_comm,"orb.imu_stamp_max_diff");
    const std::string dataset_path                      = estd2::GetStringFromYaml(conf_comm, "orb.dataset_path");
    const std::vector<int> dynamic_ids                  = LoadCategoryIds(conf_comm, "dynamic_ids");
}

void ShowParamsComm();
auto GetServerIP()->std::string;
auto GetPort()->std::string;

} //end ns
