#!/usr/bin/python3

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray, Marker
import sensor_msgs.point_cloud2 as pc2
import numpy as np

class MarkerArrayToPointcloud:
    def __init__(self):
        rospy.init_node('marker_array_to_pointcloud', anonymous=True)
        self.marker_array_sub = rospy.Subscriber("/covins_markers_be", Marker, self.marker_array_callback)
        self.pointcloud_pub1 = rospy.Publisher("/pointcloud_topic1", PointCloud2, queue_size=10)
        self.pointcloud_pub2 = rospy.Publisher("/pointcloud_topic2", PointCloud2, queue_size=10)

    def marker_array_callback(self, marker_array):
        points = []
        # print(marker_array)
        for marker in marker_array.points:
            
            points.append([marker.x, marker.y, marker.z])

        # Convert list of points to numpy array
        points_array = np.array(points, dtype=np.float32)

        # Create point cloud message
        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = marker_array.header
        pointcloud_msg.height = 1
        pointcloud_msg.width = len(points_array)
        pointcloud_msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        pointcloud_msg.is_bigendian = False
        pointcloud_msg.point_step = 12
        pointcloud_msg.row_step = pointcloud_msg.point_step * pointcloud_msg.width
        pointcloud_msg.is_dense = True
        pointcloud_msg.data = points_array.tostring()

        # Publish point cloud message
        if marker_array.ns == "Traj0_be":
            self.pointcloud_pub1.publish(pointcloud_msg)
        else:
            self.pointcloud_pub2.publish(pointcloud_msg)

if __name__ == '__main__':
    try:
        marker_array_to_pointcloud = MarkerArrayToPointcloud()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass