#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse

import cv2
import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("markers_be_topic", help="markers_be topic.")
    parser.add_argument("trajectory_topic", help="trajectory topic.")
    args = parser.parse_args()

    print(f'Extract final trajectory from {args.bag_file} on topic {args.markers_be_topic} and {args.trajectory_topic} into {args.output_dir}')

    bag = rosbag.Bag(args.bag_file, "r")
    #so only the last traj messages are read
    #get from rqt_bag. last covins_markers_be messages
    #test 6
    starttime=rospy.Time(1678452787.633855343)
    #clear files
    # f1=open(os.path.join(args.output_dir,"Traj0_be.txt"), "w")
    # f2=open(os.path.join(args.output_dir,"Traj1_be.txt"), "w")
    # f1.close()
    # f2.close()
    # f3=open(os.path.join(args.output_dir,"Traj2_be.txt"), "w")
    # f1.write(f'{bag.get_start_time()}\n')
    # f2.write(f'{bag.get_start_time()}\n')
    # f3.write(f'{bag.get_start_time()}\n')
    
    #f3.close()
    
    #Traj0_be and Traj1_be and so on    
    for topic, msg, t in bag.read_messages(topics=[args.markers_be_topic, args.trajectory_topic],start_time=starttime):
        if topic==args.markers_be_topic:
            filename=f'{msg.ns}.txt'
            f = open(os.path.join(args.output_dir,filename), "w")
            for point in msg.points:
                f.write(f'{point.x},{point.y},{point.z}\n')
            print(f'Wrote {msg.ns} points to file.')
            f.close()
        elif topic==args.trajectory_topic:
            f1=open(os.path.join(args.output_dir,"Traj0_be_time.txt"), "w")
            f2=open(os.path.join(args.output_dir,"Traj1_be_time.txt"), "w")
            trajf1=open(os.path.join(args.output_dir,"Traj0_be.txt"), "r")
            trajf2=open(os.path.join(args.output_dir,"Traj1_be.txt"), "r")
            #this works only with 2 agents
            idx=0
            for idx, time in enumerate(msg.timestamps):
                if idx < msg.sizes[0]/2:
                    pointStr=trajf1.readline()
                    f1.write(f'{pointStr.strip()},{time}\n')
                else:
                    pointStr=trajf2.readline()
                    f2.write(f'{pointStr.strip()},{time}\n')
            f1.close()
            f2.close()
            trajf1.close()
            trajf2.close()
            break
                
            
        
    bag.close()

    return

if __name__ == '__main__':
    main()