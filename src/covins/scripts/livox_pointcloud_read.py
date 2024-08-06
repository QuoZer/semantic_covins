#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class reader():
    def __init__(self):
        rospy.init_node('pointcloud_reader', anonymous=True)
        rospy.Subscriber("/covins_cloud_be", PointCloud2, self.callback)
        rospy.spin()

    def callback(self, msg):
        pc = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))
        for p in pc:
            print(p)

if __name__ == '__main__':
    reader()