# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Example to show how to use Boston Dynamics' joint control API"""

import argparse
import sys
import time
import typing
import logging
from enum import IntEnum
from multiprocessing import Barrier, Process, Queue, Value
from queue import Empty, Full
from threading import BrokenBarrierError, Thread

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
import copy
import io

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import Imu as ImuMsg

# from constants import DEFAULT_LEG_K_Q_P, DEFAULT_LEG_K_QD_P, DOF
# from joint_api_helper import JointAPIInterface

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.mission.client import MissionClient
from bosdyn.client.robot import Robot
from bosdyn.client.robot_command import (RobotCommandClient, RobotCommandStreamingClient,
                                         blocking_stand)
from bosdyn.api.spot import spot_constants_pb2
from bosdyn.client.robot_state import RobotStateStreamingClient
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks, AsyncPeriodicGRPCTask
from bosdyn.client import (
    ResponseError,
    RpcError,
    create_standard_sdk,
    frame_helpers,
    math_helpers,
)
from bosdyn.api import image_pb2, trajectory_pb2
from bosdyn.api.image_pb2 import ImageSource
from bosdyn.client.image import ImageClient
from bosdyn.client.frame_helpers import (GROUND_PLANE_FRAME_NAME, VISION_FRAME_NAME, get_a_tform_b,
                                         get_vision_tform_body,GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME)

from google.protobuf.timestamp_pb2 import Timestamp


# Link index and order
class DOF(IntEnum):
    FL_HX = spot_constants_pb2.JOINT_INDEX_FL_HX
    FL_HY = spot_constants_pb2.JOINT_INDEX_FL_HY
    FL_KN = spot_constants_pb2.JOINT_INDEX_FL_KN
    FR_HX = spot_constants_pb2.JOINT_INDEX_FR_HX
    FR_HY = spot_constants_pb2.JOINT_INDEX_FR_HY
    FR_KN = spot_constants_pb2.JOINT_INDEX_FR_KN
    HL_HX = spot_constants_pb2.JOINT_INDEX_HL_HX
    HL_HY = spot_constants_pb2.JOINT_INDEX_HL_HY
    HL_KN = spot_constants_pb2.JOINT_INDEX_HL_KN
    HR_HX = spot_constants_pb2.JOINT_INDEX_HR_HX
    HR_HY = spot_constants_pb2.JOINT_INDEX_HR_HY
    HR_KN = spot_constants_pb2.JOINT_INDEX_HR_KN
    # Arm
    A0_SH0 = spot_constants_pb2.JOINT_INDEX_A0_SH0
    A0_SH1 = spot_constants_pb2.JOINT_INDEX_A0_SH1
    A0_EL0 = spot_constants_pb2.JOINT_INDEX_A0_EL0
    A0_EL1 = spot_constants_pb2.JOINT_INDEX_A0_EL1
    A0_WR0 = spot_constants_pb2.JOINT_INDEX_A0_WR0
    A0_WR1 = spot_constants_pb2.JOINT_INDEX_A0_WR1
    # Hand
    A0_F1X = spot_constants_pb2.JOINT_INDEX_A0_F1X

    # DOF count for strictly the legs.
    N_DOF_LEGS = 12
    # DOF count for all DOF on robot (arms and legs).
    N_DOF = 19
# Default all joint gains
DEFAULT_K_Q_P = [0] * DOF.N_DOF
DEFAULT_K_QD_P = [0] * DOF.N_DOF

LOGGER = bosdyn.client.util.get_logger()

SHUTDOWN_FLAG = Value('i', 0)

# Don't let the queues get too backed up
QUEUE_MAXSIZE = 10

# This is a multiprocessing.Queue for communication between the main process and the
# Pytorch processes.
# Entries in this queue are in the format:

# {
#     'source': Name of the camera,
#     'world_tform_cam': transform from VO to camera,
#     'world_tform_gpe':  transform from VO to ground plane,
#     'raw_image_time': Time when the image was collected,
#     'cv_image': The decoded image,
#     'visual_dims': (cols, rows),
#     'depth_image': depth image proto,
#     'system_cap_time': Time when the image was received by the main process,
#     'image_queued_time': Time when the image was done preprocessing and queued
# }
RAW_IMAGES_QUEUE = Queue(QUEUE_MAXSIZE)
CAMERA_PROTO_QUEUE = Queue(QUEUE_MAXSIZE)

# Mapping from visual to depth data
VISUAL_SOURCE_TO_DEPTH_MAP_SOURCE = {
    'frontleft_fisheye_image': 'frontleft_depth_in_visual_frame',
    'frontright_fisheye_image': 'frontright_depth_in_visual_frame',
    'hand_color_image' : 'hand_depth_in_hand_color_frame'
}

ROTATION_ANGLES = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180,
    'hand_color_image' :0
}


def robotToLocalTime(timestamp: Timestamp, robot: Robot) -> Timestamp:
    """Takes a timestamp and an estimated skew and return seconds and nano seconds in local time

    Args:
        timestamp: google.protobuf.Timestamp
        robot: Robot handle to use to get the time skew
    Returns:
        google.protobuf.Timestamp
    """

    rtime = Timestamp()

    rtime.seconds = timestamp.seconds - robot.time_sync.endpoint.clock_skew.seconds
    rtime.nanos = timestamp.nanos - robot.time_sync.endpoint.clock_skew.nanos
    if rtime.nanos < 0:
        rtime.nanos = rtime.nanos + int(1e9)
        rtime.seconds = rtime.seconds - 1

    # Workaround for timestamps being incomplete
    if rtime.seconds < 0:
        rtime.seconds = 0
        rtime.nanos = 0

    return rtime


class AsyncImage(AsyncPeriodicGRPCTask):
    """
        Grab image.
    """
    def __init__(self, image_client, image_sources,sleep_between_captures):
        # Period is set to be about 15 FPS
        # super(AsyncImage, self).__init__('images', image_client, LOGGER, period_sec=1/30)
        super(AsyncImage, self).__init__(1/30) #period sec parameter
        self._query_name = 'images'
        self._client = image_client
        self._logger = LOGGER
        self._proto = None
        self.image_sources = image_sources
        self.tmp_future = None
        self.last_time = 0
        self.sleep = sleep_between_captures

    def _start_query(self):
        retval = self._client.get_image_from_sources_async(self.image_sources)
        self.tmp_future = retval
        return retval
    
    def _handle_result(self, result):
        if self.tmp_future is not None:
            start = time.time()
            self._proto = (self.tmp_future._value_from_response(self.tmp_future.original_future.result()))
            if not self._proto:
                print('continuing')
                return
            depth_responses = {
                img.source.name: img
                for img in self._proto
                if img.source.image_type == ImageSource.IMAGE_TYPE_DEPTH
            }
            image_responses = [
                img 
                for img in self._proto
                if img.source.image_type == ImageSource.IMAGE_TYPE_VISUAL
            ]
            camera_responses = {
                'image_responses' : image_responses,
                'depth_responses' : depth_responses
            }
            try:
                CAMERA_PROTO_QUEUE.put_nowait(camera_responses)
            except Full as exc:
                print(f'CAMERA_PROTO_QUEUE is full: {exc}')

    def _handle_error(self, exception):
        """Log exception.

        Args:
            exception: Error exception to log.
        """
        self._logger.exception("Failure getting %s: %s", self._query_name, exception)
    
    def _should_query(self, now_sec):
        # print("now_sec: ", now_sec, "last_call: ", self._last_call, "(now_sec - self._last_call) ",(now_sec - self._last_call),"period_sec: ", self._period_sec, "\n")     
        return super()._should_query(now_sec) 

    @property
    def proto(self):
        """Get latest response proto."""
        return self._proto

def get_source_list(image_client,onboard):
    """Gets a list of image sources and filters based on config dictionary

    Args:
        image_client: Instantiated image client
    """

    # We are using only the visual images with their corresponding depth sensors
    sources = image_client.list_image_sources()
    source_list = []
    for source in sources:
        if source.image_type == ImageSource.IMAGE_TYPE_VISUAL:
            # only append if sensor has corresponding depth sensor
            if source.name in VISUAL_SOURCE_TO_DEPTH_MAP_SOURCE:
                if onboard:
                    if source.name=="hand_color_image":
                        source_list.append(source.name)
                        source_list.append(VISUAL_SOURCE_TO_DEPTH_MAP_SOURCE[source.name])
                else:
                    source_list.append(source.name)
                    source_list.append(VISUAL_SOURCE_TO_DEPTH_MAP_SOURCE[source.name])
    return source_list


def capture_images(sleep_between_capture):
    """ Captures images and places them on the queue

    Args:
        image_task (AsyncImage): Async task that provides the images response to use
        sleep_between_capture (float): Time to sleep between each image capture
    """
    while not SHUTDOWN_FLAG.value:
        start_time = time.time()
        try:
            camera_responses = CAMERA_PROTO_QUEUE.get_nowait()
            image_responses = camera_responses['image_responses']
            depth_responses = camera_responses['depth_responses']
        except Empty:
            print('IMAGE_PROTO_QUEUE is empty')
        if not camera_responses:
            continue
        entry = {}
        for im_resp in image_responses:
            source = im_resp.source.name
            depth_source = VISUAL_SOURCE_TO_DEPTH_MAP_SOURCE[source]
            depth_image = depth_responses[depth_source]

            acquisition_time = im_resp.shot.acquisition_time
            image_time = acquisition_time.seconds + acquisition_time.nanos * 1e-9

            try:
                image = Image.open(io.BytesIO(im_resp.shot.image.data))
                image = np.asanyarray(image)
                source = im_resp.source.name

                image = ndimage.rotate(image, ROTATION_ANGLES[source])
                if source =='hand_color_image':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Converted to RGB for pytorch
                if im_resp.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Converted to RGB for TF
                tform_snapshot = im_resp.shot.transforms_snapshot
                frame_name = im_resp.shot.frame_name_image_sensor
                world_tform_cam = get_a_tform_b(tform_snapshot, VISION_FRAME_NAME, frame_name)
                world_tform_gpe = get_a_tform_b(tform_snapshot, VISION_FRAME_NAME,
                                                GROUND_PLANE_FRAME_NAME)
                entry[source] = {
                    'source': source,
                    'world_tform_cam': world_tform_cam,
                    'world_tform_gpe': world_tform_gpe,
                    'raw_image_time': image_time,
                    'cv_image': image,
                    'visual_dims': (im_resp.shot.image.cols, im_resp.shot.image.rows),
                    'depth_image': depth_image,
                    'system_cap_time': start_time,
                    'image_queued_time': time.time()
                }
            except Exception as exc:  # pylint: disable=broad-except
                print(f'Exception occurred during image capture {exc}')
        try:
            RAW_IMAGES_QUEUE.put_nowait(entry)
        except Full as exc:
            print(f'RAW_IMAGES_QUEUE is full: {exc}')
        time.sleep(sleep_between_capture)


class AsyncState:
    '''
        Get the robot minimal robot state stream 
    '''
    
    def __init__(self, robot, n_dofs):
        self.robot = robot
        self.latest_state_stream_data = None
        self.should_stop = False
        self.started_streaming = False
        self.cmd_history = {}

        if n_dofs != DOF.N_DOF_LEGS and n_dofs != DOF.N_DOF:
            # DOF error
            self.robot.logger.warning("Incorrect number of DOF. Joint API will not be activated")
            self.should_stop = True
        self.n_dofs = n_dofs

    def handle_state_streaming(self, robot_state_streaming_client):
        for state in robot_state_streaming_client.get_robot_state_stream():
            receive_time = time.time()
            self.latest_state_stream_data = state

            if self.should_stop:
                return

            if (state.last_command.user_command_key != 0 and
                    state.last_command.user_command_key in self.cmd_history.keys()):
                sent_time = self.cmd_history[state.last_command.user_command_key]
                received_time = self.robot.time_sync.get_robot_time_converter(
                ).local_seconds_from_robot_timestamp(state.last_command.received_timestamp)
                latency_ms = (received_time - sent_time) * 1000
                roundtrip_ms = (receive_time - sent_time) * 1000
                # Note that the latency measurement here has the receive time converted from
                # robot time, so it's not very accurate
                self.robot.logger.info(f"Latency {latency_ms:.3f}\tRoundtrip {roundtrip_ms:.3f}")
            else:
                self.robot.logger.info(f"No key: {state.last_command.user_command_key}")

    # get_latest_joints_state function is to get a latest joint state
    def get_latest_imu_state(self):
        # Wait for first data to cache. This should happend synchronously in normal stand before
        # joint control is activated.
        while not self.latest_state_stream_data:
            time.sleep(0.1)

        imu_state = self.latest_state_stream_data.inertial_state

        return imu_state
    
    def set_should_stop(self, val):
        self.should_stop = val
    
### SPOT WRAPPER CLASSES 
    
class ThreadedFunctionLoop:
    """
    Holds a thread which calls a function in a loop at the specified rate
    """

    def __init__(self, function: typing.Callable, rate: rospy.Rate) -> None:
        """
        Args:
            function: Function which should be called at the specified rate
            rate: Rate at which the function should be called
        """

        self.function = function
        self.rate = rate
        self.thread = Thread(target=self.run)
        self.thread.start()

    def run(self):
        while not rospy.is_shutdown():
            self.function()
            self.rate.sleep()

class RateLimitedCall:
    """
    Wrap a function with this class to limit how frequently it can be called within a loop
    """

    def __init__(self, fn: typing.Callable, rate_limit: float):
        """

        Args:
            fn: Function to call
            rate_limit: The function will not be called faster than this rate
        """
        self.fn = fn
        self.min_time_between_calls = 1.0 / rate_limit
        self.last_call = 0

    def __call__(self):
        now_sec = time.time()
        if (now_sec - self.last_call) > self.min_time_between_calls:
            self.fn()
            self.last_call = now_sec

class ClientWrapper:
    '''
        ROS Wrapper 
    '''
    def __init__(self):
        rospy.init_node('spot_client', anonymous=True)

        self.bridge = CvBridge()
        self.rate = rospy.Rate(50)
        self._logger = logging.getLogger("rosout")

        self.robot_name = rospy.get_param("~robot_name", "spot")
        self._username = rospy.get_param("~username", "user")
        self._password = rospy.get_param("~password", "ruy6ucvnoj4z")
        self._hostname = rospy.get_param("~hostname", "192.168.80.3")

        self.camera_name = rospy.get_param("~camera_name", "frontleft")

        try:
            self._sdk = create_standard_sdk(
                'spot_client', service_clients=[MissionClient], cert_resource_glob=None
            )
        except Exception as e:
            self._logger.error("Error creating SDK object: %s", e)
            self._valid = False
            return
        # self._sdk.register_service_client(RobotStateStreamingClient)          <--  requires a special license 
        # self._sdk.register_service_client(RobotCommandStreamingClient)

        self._robot = self._sdk.create_robot(self._hostname)


        authenticated = self.authenticate(self._robot, self._username, self._password, self._logger)

        self.init_publishers()

    def init_publishers(self):
        self.img_pub = rospy.Publisher("/spot/image", ImageMsg, queue_size=1)
        self.imu_pub = rospy.Publisher("/spot/imu", ImuMsg, queue_size=1)

    def authenticate(self, robot: Robot, username: str, password: str, logger: logging.Logger) -> bool:
        """
        Authenticate with a robot through the bosdyn API. A blocking function which will wait until authenticated (if
        the robot is still booting) or login fails

        Args:
            robot: Robot object which we are authenticating with
            username: Username to authenticate with
            password: Password for the given username
            logger: Logger with which to print messages

        Returns:
            boolean indicating whether authentication was successful
        """
        authenticated = False
        while not authenticated:
            try:
                logger.info("Trying to authenticate with robot...")
                robot.authenticate(username, password)
                robot.time_sync.wait_for_sync(10)
                logger.info("Successfully authenticated.")
                authenticated = True
            except RpcError as err:
                sleep_secs = 15
                logger.warn(
                    "Failed to communicate with robot: {}\nEnsure the robot is powered on and you can "
                    "ping {}. Robot may still be booting. Will retry in {} seconds".format(
                        err, robot.address, sleep_secs
                    )
                )
                time.sleep(sleep_secs)
            except bosdyn.client.auth.InvalidLoginError as err:
                logger.error("Failed to log in to robot: {}".format(err))
                raise err

        return authenticated

    def publish_images(self):
        try: 
            entry = RAW_IMAGES_QUEUE.get_nowait()
        except Empty:
            return
        
        for _, capture in entry.items():
            image = capture['cv_image']
            img_msg = self.bridge.cv2_to_imgmsg(image, "passthrough")
            self.img_pub.publish(img_msg)

    def publish_imu(self):
        state = self.state_interface.get_latest_imu_state()
        last_packet = state.packets[-1] # publishing only one atm as it would break the rates otherwise 
        
        imu_msg = ImuMsg()
        imu_msg.header.frame_id = state.mounting_link_name
        local_time = robotToLocalTime(last_packet.timestamp, self.robot)
        imu_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)

        imu_msg.linear_acceleration = last_packet.acceleration_rt_odom_in_link_frame
        imu_msg.angular_velocity = last_packet.angular_velocity_rt_odom_in_link_frame
        imu_msg.orientation = last_packet.odom_rot_link

        self.imu_pub.publish(imu_msg)



    def _update_thread(self, async_task):
        while True:
            async_task.update()
            time.sleep(0.01)


    def run(self):
        # Establish time sync with the robot. 
        self._robot.time_sync.wait_for_sync()

        # Verify the robot is not estopped
        assert not self._robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                        'such as the estop SDK example, to configure E-Stop.'
        print("Acquiring lease") 
        # Acquire the lease
        lease_client = self._robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            # Now, we are ready to power on the robot. 
            self._logger.info('Powering on robot... This may take several seconds.')
            self._robot.power_on(timeout_sec=20)
            assert self._robot.is_powered_on(), 'Robot power on failed.'
            self._logger.info('Robot powered on.')

            self._logger.info("Creating clients...")
            initialised = False
            while not initialised:
                try:
                    # The robot state streaming client will allow us to get the robot's joint and imu information.
                    # robot_state_streaming_client = self._robot.ensure_client(RobotStateStreamingClient.default_service_name)
                    command_client = self._robot.ensure_client(RobotCommandClient.default_service_name)
                    image_client = self._robot.ensure_client(ImageClient.default_service_name)
                   
                    source_list = get_source_list(image_client,False)
                    image_task = AsyncImage(image_client, source_list,1/30)
                    task_list = [image_task]
                    _async_tasks = AsyncTasks(task_list)
                    self._logger.info('Detect and follow client connected.')
                    initialised = True
                except Exception as e:
                    sleep_secs = 15
                    self._logger.warning(
                        "Unable to create client service: {}. This usually means the robot hasn't "
                        "finished booting yet. Will wait {} seconds and try again.".format(e, sleep_secs)
                    )
                    time.sleep(sleep_secs)


            # Stand the robot
            blocking_stand(command_client)
            # Extra delay to make sure
            time.sleep(2)        

            self.state_interface = AsyncState(self._robot, DOF.N_DOF_LEGS)

            img_rates = 10
            state_rates = 50
            rate_limited_camera_images = RateLimitedCall(
                self.publish_images,  max(0.0, img_rates),
            )
            # rate_limited_state = RateLimitedCall(
            #     self.publish_imu, state_rates
            # )


            state_thread = None
            try:
                # Start state streaming
                # self._logger.info("Starting state stream")
                # state_thread = Thread(target=self.state_interface.handle_state_streaming,
                #                     args=(robot_state_streaming_client,))
                # state_thread.start()

                # This thread starts the async tasks for image  retrieval
                update_thread = Thread(target=self._update_thread, args=[_async_tasks])
                update_thread.daemon = True
                update_thread.start()

                # Publishing threads 
                camera_publish_thread = ThreadedFunctionLoop(rate_limited_camera_images, self.rate)
                # state_publish_thread = ThreadedFunctionLoop(rate_limited_state, self.rate)


            finally:
                self.state_interface.set_should_stop(True)
                if state_thread:
                    state_thread.join()

            # Power the robot off. By specifying "cut_immediately=False", a safe power off command
            # is issued to the robot. This will attempt to sit the robot before powering off.
            self._robot.power_off(cut_immediately=False, timeout_sec=20)
            assert not self._robot.is_powered_on(), 'Robot power off failed.'
            self._logger.info('Robot safely powered off.')


def main():
    """Command line interface."""
    
    client_node = ClientWrapper()
    print("Wrapper created")
    try:
        client_node.run()    
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = client_node._logger
        logger.exception('Hello, Spot! threw an exception: %r', exc)
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)