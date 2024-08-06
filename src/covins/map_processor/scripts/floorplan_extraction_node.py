#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
import message_filters

from ctypes import *
import struct
import numpy as np
import open3d as o3d
# import ros_numpy
from mapper.mapper import Mapper 

name_to_dtypes = {
	"rgb8":    (np.uint8,  3),
	"rgba8":   (np.uint8,  4),
	"rgb16":   (np.uint16, 3),
	"rgba16":  (np.uint16, 4),
	"bgr8":    (np.uint8,  3),
	"bgra8":   (np.uint8,  4),
	"bgr16":   (np.uint16, 3),
	"bgra16":  (np.uint16, 4),
	"mono8":   (np.uint8,  1),
	"mono16":  (np.uint16, 1),
	
    # for bayer image (based on cv_bridge.cpp)
	"bayer_rggb8":	(np.uint8,  1),
	"bayer_bggr8":	(np.uint8,  1),
	"bayer_gbrg8":	(np.uint8,  1),
	"bayer_grbg8":	(np.uint8,  1),
	"bayer_rggb16":	(np.uint16, 1),
	"bayer_bggr16":	(np.uint16, 1),
	"bayer_gbrg16":	(np.uint16, 1),
	"bayer_grbg16":	(np.uint16, 1),

    # OpenCV CvMat types
	"8UC1":    (np.uint8,   1),
	"8UC2":    (np.uint8,   2),
	"8UC3":    (np.uint8,   3),
	"8UC4":    (np.uint8,   4),
	"8SC1":    (np.int8,    1),
	"8SC2":    (np.int8,    2),
	"8SC3":    (np.int8,    3),
	"8SC4":    (np.int8,    4),
	"16UC1":   (np.uint16,   1),
	"16UC2":   (np.uint16,   2),
	"16UC3":   (np.uint16,   3),
	"16UC4":   (np.uint16,   4),
	"16SC1":   (np.int16,  1),
	"16SC2":   (np.int16,  2),
	"16SC3":   (np.int16,  3),
	"16SC4":   (np.int16,  4),
	"32SC1":   (np.int32,   1),
	"32SC2":   (np.int32,   2),
	"32SC3":   (np.int32,   3),
	"32SC4":   (np.int32,   4),
	"32FC1":   (np.float32, 1),
	"32FC2":   (np.float32, 2),
	"32FC3":   (np.float32, 3),
	"32FC4":   (np.float32, 4),
	"64FC1":   (np.float64, 1),
	"64FC2":   (np.float64, 2),
	"64FC3":   (np.float64, 3),
	"64FC4":   (np.float64, 4)
}
# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}

convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
def convert_rgbFloat_to_tuple(rgb_float): 
    # # cast float32 to int so that bitwise operations are possible
    # s = struct.pack('>f' ,rgb_float)
    # i = struct.unpack('>l',s)[0]
    # # you can get back the float value by the inverse operations
    # pack = c_uint32(i).value
    pack =int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
    
    rgb_int = convert_rgbUint32_to_tuple(pack)
    
    return rgb_int

class FloorplanExtractor:
    def __init__(self):
        rospy.init_node('floorplan_extraction_node', anonymous=True)

        dataset_path = rospy.get_param('~dataset_path')

        # Initialize Mapper
        self.mapper = Mapper(vis=False)
        self.mapper.load_dataset(dataset_path)

        # State
        self.state = 'WAITING'
        self.raw_map = None
        self.processed_map = None
        self.map_frame = 'world'

        # Subscribers
        raw_map_sub = message_filters.Subscriber('/slam/raw_map', PointCloud2)
        processed_map_sub = message_filters.Subscriber('/slam/esdf', PointCloud2)

        raw_map_sub.registerCallback(self.ptc_callback)
        processed_map_sub.registerCallback(self.esdf_callback)

        # Synchronize raw and processed map messages
        # ts = message_filters.ApproximateTimeSynchronizer([raw_map_sub, processed_map_sub], 10, 10)
        # ts.registerCallback(self.map_callback)

        # Subscribe to post-processing completion signal
        rospy.Subscriber('/slam/state', Bool, self.post_processing_complete_callback)

        # Publisher for the final floorplan image
        self.floorplan_pub = rospy.Publisher('/floorplan', Image, queue_size=10)

    def map_callback(self, raw_map_msg, processed_map_msg):
        rospy.logwarn("Received maps")
        # Store the latest maps
        self.raw_map = raw_map_msg
        self.processed_map = processed_map_msg

    def esdf_callback(self, msg):
        # rospy.logwarn("Received esdf")
        # Store the latest maps
        self.processed_map = msg

    def ptc_callback(self, msg):
        rospy.logwarn("Received raw map")
        # Store the latest maps
        self.raw_map = msg

    def post_processing_complete_callback(self, msg):
        if msg.data and self.state == 'WAITING':
            self.state = 'PROCESSING'
            self.process_maps()

    def process_maps(self):
        if self.raw_map is None or self.processed_map is None:
            rospy.logwarn("Raw or processed map not available")
            self.state = 'WAITING'
            return

        # Convert PointCloud2 messages to numpy arrays
        # pcd = self.pointcloud2_to_array(self.raw_map, 'xyzrgb')
        # esdf = self.pointcloud2_to_array(self.processed_map, 'xyzi')
        pcd = self.convertCloudFromRosToOpen3d(self.raw_map)
        esdf = self.convertCloudFromRosToOpen3d(self.processed_map)

        # Load maps into Mapper
        self.mapper.setup_point_clouds(pcd, "Map", 0, False)
        self.mapper.setup_intensity_map(esdf, "ESDF", 0, False)

        self.mapper.m2d.set_resolution(0.2, 60, -2)
        floorplan_image = self.mapper.m2d.fuse_pointclouds([0,0,0])
        # floorplan_image = self.mapper.class_colors[floorplan_idx]

        # Convert numpy array to ROS Image message
        image_msg = self.numpy_to_image_msg(floorplan_image, encoding="rgba8")

        # Publish the floorplan image
        self.floorplan_pub.publish(image_msg)
        self.state = 'WAITING'

    def convertCloudFromRosToOpen3d(self, ros_cloud):
        # https://github.com/felixchenfy/open3d_ros_pointcloud_conversion/blob/master/lib_cloud_conversion_between_Open3D_and_ROS.py
        # Get cloud data from ros_cloud
        field_names=[field.name for field in ros_cloud.fields]
        print(field_names)
        cloud_data = list(pc2.read_points(ros_cloud, skip_nans=True, field_names=field_names))

        # Check empty
        open3d_cloud = o3d.geometry.PointCloud()
        if len(cloud_data)==0:
            print("Converting an empty cloud")
            return None

        # Set open3d_cloud
        if "rgb" in field_names:
            IDX_RGB_IN_FIELD=3 # x, y, z, rgb
            
            # Get xyz  HACK 
            xyz = [(row[0],row[1],row[2]) for row in cloud_data ] # (why cannot put this line below rgb?)

            # Get rgb
            # Check whether int or float
            if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
                rgb = [convert_rgbFloat_to_tuple(row[IDX_RGB_IN_FIELD]) for row in cloud_data ]
            else:
                rgb = [convert_rgbUint32_to_tuple(row[IDX_RGB_IN_FIELD]) for row in cloud_data ]

            # combine
            open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
            open3d_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb)/255.0)
        elif 'intensity' in field_names:
            xyz = [(row[0],row[1],row[2]) for row in cloud_data ] # get xyz
            rgb = [(row[3],0,0) for row in cloud_data ] 
            open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
            open3d_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb))
        elif 'label' in field_names:
            xyz = [(row[0],row[1],row[2]) for row in cloud_data ]
            rgb = [self.mapper.class_colors[row[3]+1] for row in cloud_data ]  # msg labels start from 0, here from 1 
            # print(np.array(rgb)[10:30])
            open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
            open3d_cloud.colors = o3d.utility.Vector3dVector( (255-np.array(rgb))/255.0)

        # return
        return open3d_cloud

    def fields_to_dtype(self, fields, point_step):
        '''Convert a list of PointFields to a numpy record datatype.
        '''
        offset = 0
        np_dtype_list = []
        for f in fields:
            while offset < f.offset:
                # might be extra padding between fields
                np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
                offset += 1

            dtype = pftype_to_nptype[f.datatype]
            if f.count != 1:
                dtype = np.dtype((dtype, f.count))

            np_dtype_list.append((f.name, dtype))
            offset += pftype_sizes[f.datatype] * f.count

        # might be extra padding between points
        while offset < point_step:
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1
            
        return np_dtype_list

    def pointcloud2_to_array(self, cloud_msg, cloud_type):
        # Convert PointCloud2 message to numpy array
        # construct a numpy record type equivalent to the point type of this cloud
        print(cloud_msg.fields, cloud_msg.point_step)
        dtype_list = self.fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

        # parse the cloud into an array
        print(len(cloud_msg.data), dtype_list) #392
        cloud_arr = np.frombuffer(cloud_msg.data, dtype_list) 
        # remove the dummy fields that were added
        cloud_arr = cloud_arr[
            [fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]
        
        pc = np.reshape(cloud_arr, (cloud_msg.width,))

        points=np.zeros( (pc.shape[0],len(cloud_type)) )
        for i, dim in enumerate(cloud_type):
            points[:,i]=pc[dim]

        return points


    def numpy_to_image_msg(self, image_array, encoding):
        # Convert numpy array to ROS Image message (from ros_numpy pkg)
        im = Image(encoding=encoding)

        # extract width, height, and channels
        dtype_class, exp_channels = name_to_dtypes[encoding]
        dtype = np.dtype(dtype_class)
        if len(image_array.shape) == 2:
            im.height, im.width, channels = image_array.shape + (1,)
        elif len(image_array.shape) == 3:
            im.height, im.width, channels = image_array.shape
        else:
            raise TypeError("Array must be two or three dimensional")

        # check type and channels
        if exp_channels != channels:
            raise TypeError("Array has {} channels, {} requires {}".format(
                channels, encoding, exp_channels
            ))
        if dtype_class != image_array.dtype.type:
            raise TypeError("Array is {}, {} requires {}".format(
                image_array.dtype.type, encoding, dtype_class
            ))

        # make the array contiguous in memory, as mostly required by the format
        contig = np.ascontiguousarray(image_array)
        im.data = contig.tostring()
        im.step = contig.strides[0]
        im.is_bigendian = (
            image_array.dtype.byteorder == '>' or 
            image_array.dtype.byteorder == '=' and sys.byteorder == 'big'
        )

        return im

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = FloorplanExtractor()
        node.run()
    except rospy.ROSInterruptException:
        pass