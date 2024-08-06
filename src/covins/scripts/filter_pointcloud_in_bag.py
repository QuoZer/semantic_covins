# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import rosbag
import rospy
import pcl
import ctypes
import struct
from roslib import message
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def main():
    bag = rosbag.Bag("/home/deva/work/COVINS/bags/09_23_recording_test_1663922957_cut_2.bag", 'r')
    bag2 = rosbag.Bag("/home/deva/work/COVINS/bags/test.bag", 'w')
    for topic, msg, t in bag.read_messages(topics="/covins_cloud_be"):
        pc = pc2.read_points(msg)
        points_list = []  
        for p in pc:
            test = p[3] 
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack('>f' ,test)
            i = struct.unpack('>l',s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000)>> 16
            g = (pack & 0x0000FF00)>> 8
            b = (pack & 0x000000FF)
            #print(r,g,b)
            #print(p[0],p[1],p[2])
            if p[0]<8.0 and p[0]>4.0:
                if p[1]>-6.0 and p[1]<-2.0:
                    if p[2]>-0.5 and p[2]<2.0:
                        points_list.append(p)                   
        XYZRGB_cloud = pc2.create_cloud(msg.header, msg.fields, points_list)
        bag2.write("/covins_cloud_be",XYZRGB_cloud)


    bag.close()
    bag2.close()
    return

if __name__ == '__main__':
    main()