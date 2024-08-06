#!/usr/bin/python

import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu

import PIL.Image
import cv2
import numpy as np
import math
import yaml

import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from torch2trt import TRTModule
import tensorrt as trt

from nanosam.utils.predictor import Predictor
from nanosam.utils.owlvit import OwlVit
from efficientvit.seg_model_zoo import create_seg_model

#from jetson_inference import segNet


class ToTensor(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, img):
        image = img
        image = image.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
        image = torch.as_tensor(image, dtype=torch.float32).div(255.0)
        image = F.normalize(image, self.mean, self.std, self.inplace)
        return image

class SAMNode:
    def __init__(self, debug=False):
        self.node_initialized = False
        self.debug = debug

        rospy.init_node('segment_anything_node', anonymous=True)
        self.bridge = CvBridge()

        # Params 
        dataset_objects = rospy.get_param('~dataset_path', None)
        method = rospy.get_param('~segmentation_method', 'vit')
        engine_path = rospy.get_param('~model_engine_path', None)
        weights_path = rospy.get_param('~model_weights_path', None)
        synchronize_outputs = rospy.get_param('~synchronize_outputs', False)
        
        self.load_dataset(dataset_objects)

        # Publish masks 
        self.mask_publisher = rospy.Publisher("/sam_node/mask", Image, queue_size=1)
        # 
        self.imu_sub0 = rospy.Subscriber("/camera/imu0", Imu, self.delay_imu)
        self.imu_delayer = rospy.Publisher("/sam_node/dt_imu", Imu, queue_size=1)
        self.image_publisher = rospy.Publisher("/sam_node/image", Image, queue_size=1)
        if synchronize_outputs:
            # Input images
            self.image_subscriber = message_filters.Subscriber("/camera/image_raw", Image)
            self.imu_subscriber = message_filters.Subscriber("/camera/imu", Imu)
            self.imu_publisher = rospy.Publisher("/sam_node/imu", Imu, queue_size=1)

            ts = message_filters.ApproximateTimeSynchronizer([self.image_subscriber, self.imu_subscriber], 10, 0.01)
            ts.registerCallback(self.synchro_cb)
        else:
            # Input images
            self.image_subscriber = rospy.Subscriber("/camera/image_raw", Image, self.image_cb, queue_size=1)    

        self.method = method
        print("Chosen method: ", method)
        if method == "sam" or method == "owl":
            # Instantiate TensorRT predictor (TODO: make a param)
            self.predictor = Predictor(
                "/home/appuser/data/shared_data/resnet18_image_encoder.engine",
                "/home/appuser/data/shared_data/mobile_sam_mask_decoder.engine"
            )
            self.image_size = np.array([[640, 480]])
        if method == "owl":
            # vit detector (threshold)
            self.detector = OwlVit(0.1)
            self.prompt = "Table"
        if method == "vit":
            if engine_path:
                self.predictor = self.load_vit_engine(engine_path)
            elif weights_path:
                self.predictor = create_seg_model( name="l1", dataset="ade20k", weight_url=weights_path ).cuda().eval() # TODO: param
            self.crop_size = 512
            self.transform = transforms.Compose(
            [
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        if method == "segnet":
            self.predictor = segNet("fcn-resnet18-sun-640x512")

        self.node_initialized = True

    def load_dataset(self, yaml_file:str) -> None:
        '''
            Parse the object labels and colors from a .yaml file
        '''
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
        
        colors = data['colors']
        labels = data['labels']
        
        color_label_map = {}
        for idx, color in enumerate(colors):
            color_label_map[labels[idx]] = np.array(color)
        self.dataset = color_label_map
        
        # Append the void class 
        self.class_colors = np.array(colors)
        self.class_labels = np.array(labels)
        
    # Callback for the subscribed topic 
    def image_cb(self, data):
        if not self.node_initialized:
            return
        # start_time = rospy.Time.now()
        # Convert the img
        try: 
            cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except Exception as e:
            rospy.logerr(e)
            return
        if len(cv_image.shape) == 2: # one channel
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

        if self.method == "sam":
            masked_img = self.segment_simple(cv_image)
        if self.method == "owl":
            masked_img = self.segment_owl(cv_image)
        if self.method == "vit":
            masked_img = self.segment_vit(cv_image)

        msg_mask = self.bridge.cv2_to_imgmsg(masked_img, "passthrough")
        # end_time = rospy.Time.now()
        # dt = end_time - start_time
        # print(f"Duration: {dt.secs}s {dt.nsecs/1000000}ms")
        # Publish
        self.mask_publisher.publish(msg_mask)   
        self.image_publisher.publish(data)

    def synchro_cb(self, image, imu):
        if not self.node_initialized:
            return

        # Convert the img
        try: 
            cv_image = self.bridge.imgmsg_to_cv2(image, "passthrough")
        except Exception as e:
            rospy.logerr(e)
            return
        if len(cv_image.shape) == 2: # one channel
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

        if self.method == "sam":
            masked_img = self.segment_simple(cv_image)
        if self.method == "owl":
            masked_img = self.segment_owl(cv_image)
        if self.method == "vit":
            masked_img = self.segment_vit(cv_image)

        # cv2.imwrite("/home/appuser/data/shared_data/mask.png", masked_img)
        msg_mask = self.bridge.cv2_to_imgmsg(masked_img, "mono8")
        
        # msg_mask.header.stamp = stamp
        # image.header.stamp = stamp
        # imu.header.stamp = stamp
        # Publish
        self.mask_publisher.publish(msg_mask)   
        self.image_publisher.publish(image)
        self.imu_publisher.publish(imu)

    def delay_imu(self, imu_msg):
        dt = rospy.Duration(0, 20435095)
        old_stamp = imu_msg.header.stamp
        new_stamp = old_stamp - dt
        imu_msg.header.stamp = new_stamp

        self.imu_delayer.publish(imu_msg)


    def segment_simple(self, cv_image):
        '''
            NanoSAM segmentation with no lables or semantics 
        '''
        pil_image = PIL.Image.fromarray(cv_image)

        self.predictor.set_image(pil_image)
        # Central 200x200 bbox
        # bbox = [*(self.image_size/2-100), *(self.image_size/2+100)]
        # points  = np.array([
        #     [bbox[0], bbox[1]],
        #     [bbox[2], bbox[3]]
        # ])
        # Predict mask
        point_labels = np.array([2, 3])
        # mask, _, _ = self.predictor.predict(points, point_labels)
        mask, _, _ = self.predictor.predict(self.image_size/2, np.array([1]))

        mask = (mask[0, 0] > 0).detach().cpu().numpy()
        masked_img = cv_image.copy()
        masked_img[mask] = 0
        
        return masked_img
    
    
    def segment_owl(self, cv_image):
        '''
            Open Vocabulary object detection and segmentation based on a text prompt. Very slow. 
        '''
        pil_image = PIL.Image.fromarray(cv_image)
        pil_image = pil_image.convert("RGB")
        # Set image
        self.predictor.set_image(pil_image)
        # Find the object
        detections = self.detector.predict(pil_image, texts=self.prompt )
        N = len(detections)
        if N == 0:
            return cv_image
        
        detections = sorted(detections, key=lambda item: item['score'])
        bbox = detections[-1]['bbox']
        print(detections)
        points  = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[3]]
        ])
        point_labels = np.array([2, 3])
        #  Segment the object
        mask, _, _ = self.predictor.predict(points, point_labels)

        mask = (mask[0, 0] > 0).detach().cpu().numpy()
        # 
        opacity = 0.5
        masked_img = cv_image.copy()
        masked_img[mask,:] = [0,0,255]
        masked_img = masked_img * opacity + cv_image * (1 - opacity)
        masked_img = np.asarray(masked_img, dtype=np.uint8)
        
        return masked_img


    def segment_vit(self, cv_image):
        '''
            Perform semantic segmentation using a Visual transformer trained on the ADE20k dataset. 
        '''
        image = cv_image
        res_image = image

        h, w = image.shape[:2]      
        # Realsense: 720x1280 (1.777) | th=512 tw=928 (1.812) || 512x704 (1.375) || 480x640
        if h < w:
            th = self.crop_size
            tw = math.ceil(w / h * th / 32) * 32
        else:
            tw = self.crop_size
            th = math.ceil(h / w * tw / 32) * 32
        if th != h or tw != w:
            res_image = cv2.resize(
                image,
                dsize=(tw, th),
                interpolation=cv2.INTER_CUBIC,
            )
        # different aspect ratio
        if tw > 704:
            dw = (tw-704)//2
            res_image = res_image[:, dw:-dw,:]

        # print((h,w), (th,tw) ,res_image.shape, res_image.dtype)
        # weird 
        # res_image = res_image.astype(np.int8)

        image_tensor = self.transform(res_image).cuda()
        image_tensor = image_tensor.unsqueeze(0)

        # Run the network
        output = self.predictor(image_tensor)

        output = torch.argmax(output, dim=1)
        pred = output[0].cpu().numpy()
        # pred = self.process_segmentation_output(output, 8, 0)

        draw_overlay = False
        if draw_overlay:
            canvas = np.asarray(pred, dtype=np.uint8)
        else:
            canvas = self.get_canvas(image, pred)

        return canvas
    
    # TODO: Segnet from jetson_inference 
    def segment_segnet(self, pil_image):

        self.predictor.predict()



    def get_canvas(self,
                    image: np.ndarray,
                    mask: np.ndarray,
                    opacity=0.5,
                ) -> np.ndarray:
        '''
            Draw a segmentation mask on top of an image
        '''

        if self.class_colors == None or self.class_labels == None:
            rospy.logerr("No dataset loaded")       # technically still can color with some random colors 
            return np.asarray(mask, dtype=np.uint8)
        
        image_shape = image.shape[:2]
        mask_shape = mask.shape
        if image_shape != mask_shape:
            mask = cv2.resize(mask, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        seg_mask = np.zeros_like(image, dtype=np.uint8)

        frame_classes = []
        for k, color in enumerate(self.class_colors):
            class_mask = mask == k
            if np.sum(class_mask) == 0:
                continue
            #print(k, np.sum(class_mask))
            frame_classes.append(k)
            seg_mask[class_mask, :] = color
        canvas = seg_mask * opacity + image * (1 - opacity)
        canvas = np.asarray(canvas, dtype=np.uint8)
        if self.debug:
            for i, c in enumerate(frame_classes):
                canvas = cv2.putText(canvas, self.class_labels[c], (10,30+20*i), cv2.FONT_HERSHEY_SIMPLEX, 1, self.class_colors[c], 1, cv2.LINE_AA)
        return canvas
    
    # A function to load a model from a tensorrt engine file 
    def load_vit_engine(self, path: str):
    
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(path, 'rb') as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)

        mask_predictor_trt = TRTModule(
            engine=engine,
            input_names=[
                "input"
            ],
            output_names=[
                "output"  # bruh
            ]
        )

        return mask_predictor_trt

if __name__ == "__main__":
    try:
        #
        node = SAMNode(debug=True,)
        print("segmentation node started")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

