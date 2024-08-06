#!/usr/bin/python3

import rospy
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg


class AxisAlignBroadcaster:

    def __init__(self):
        self.pub_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)
        self.axis_sub = rospy.Subscriber('/initialpose', geometry_msgs.msg.PoseWithCovarianceStamped, self.initial_pose_callback)

        # Set the initial pose
        initial_pose = geometry_msgs.msg.TransformStamped()
        initial_pose.header.frame_id = 'world'
        initial_pose.child_frame_id = 'odom'
        initial_pose.transform.translation.x = 0.0
        initial_pose.transform.translation.y = 0.0
        initial_pose.transform.translation.z = 0.0
        initial_pose.transform.rotation.x = 0.0
        initial_pose.transform.rotation.y = 0.0
        initial_pose.transform.rotation.z = 0.0
        initial_pose.transform.rotation.w = 1.0
        self.transform = initial_pose

        self.main_loop()

    def initial_pose_callback(self, msg):
        # Update the pose based on the orientation of the received message

        self.transform.transform.rotation.x = msg.pose.pose.orientation.x
        self.transform.transform.rotation.y = msg.pose.pose.orientation.y
        self.transform.transform.rotation.z = msg.pose.pose.orientation.z
        self.transform.transform.rotation.w = msg.pose.pose.orientation.w
        # print("Updated world->odom tf to: " + msg.pose.pose.orientation.x)

    def main_loop(self):

        while not rospy.is_shutdown():
            # Run this loop at about 10Hz
            rospy.sleep(0.1)

            tf_message = self.transform
            tf_message.header.stamp = rospy.Time.now()
            tfm = tf2_msgs.msg.TFMessage([tf_message])
            self.pub_tf.publish(tfm)

if __name__ == '__main__':
    rospy.init_node('world_odom_transform_node')
    tfb = AxisAlignBroadcaster()

    rospy.spin()
