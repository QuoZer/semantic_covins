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

import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from torch2trt import TRTModule
import tensorrt as trt

from nanosam.utils.predictor import Predictor
from nanosam.utils.owlvit import OwlVit
from efficientvit.seg_model_zoo import create_seg_model

#from jetson_inference import segNet


CLASS_COLORS = (
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    )

CLASSES = (
    "wall",
    "building",
    "sky",
    "floor",
    "tree",
    "ceiling",
    "road",
    "bed",
    "windowpane",
    "grass",
    "cabinet",
    "sidewalk",
    "person",
    "earth",
    "door",
    "table",
    "mountain",
    "plant",
    "curtain",
    "chair",
    "car",
    "water",
    "painting",
    "sofa",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug",
    "field",
    "armchair",
    "seat",
    "fence",
    "desk",
    "rock",
    "wardrobe",
    "lamp",
    "bathtub",
    "railing",
    "cushion",
    "base",
    "box",
    "column",
    "signboard",
    "chest of drawers",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace",
    "refrigerator",
    "grandstand",
    "path",
    "stairs",
    "runway",
    "case",
    "pool table",
    "pillow",
    "screen door",
    "stairway",
    "river",
    "bridge",
    "bookcase",
    "blind",
    "coffee table",
    "toilet",
    "flower",
    "book",
    "hill",
    "bench",
    "countertop",
    "stove",
    "palm",
    "kitchen island",
    "computer",
    "swivel chair",
    "boat",
    "bar",
    "arcade machine",
    "hovel",
    "bus",
    "towel",
    "light",
    "truck",
    "tower",
    "chandelier",
    "awning",
    "streetlight",
    "booth",
    "television receiver",
    "airplane",
    "dirt track",
    "apparel",
    "pole",
    "land",
    "bannister",
    "escalator",
    "ottoman",
    "bottle",
    "buffet",
    "poster",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer belt",
    "canopy",
    "washer",
    "plaything",
    "swimming pool",
    "stool",
    "barrel",
    "basket",
    "waterfall",
    "tent",
    "bag",
    "minibike",
    "cradle",
    "oven",
    "ball",
    "food",
    "step",
    "tank",
    "trade name",
    "microwave",
    "pot",
    "animal",
    "bicycle",
    "lake",
    "dishwasher",
    "screen",
    "blanket",
    "sculpture",
    "hood",
    "sconce",
    "vase",
    "traffic light",
    "tray",
    "ashcan",
    "fan",
    "pier",
    "crt screen",
    "plate",
    "monitor",
    "bulletin board",
    "shower",
    "radiator",
    "glass",
    "clock",
    "flag",
)


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
    def __init__(self, method, debug=False, synchronize=True):
        self.node_initialized = False
        self.debug = debug
        self.synchronize = synchronize


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
            #self.predictor = create_seg_model( name="l1", dataset="ade20k", weight_url="/home/appuser/workspace/efficientvit/assets/checkpoints/seg/ade20k/l1.pt" ).cuda().eval() # TODO: param
            self.predictor = self.load_vit_engine("/home/appuser/data/shared_data/evit_ade20k_l1_bs1_fp16.trt")
            self.crop_size = 512
            self.transform = transforms.Compose(
            [
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        if method == "segnet":
            self.predictor = segNet("fcn-resnet18-sun-640x512")

        self.node_initialized = True
        
        
    def segment(self, cv_image):
        if not self.node_initialized:
            return

        if len(cv_image.shape) == 2: # one channel
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

        if self.method == "sam":
            masked_img = self.segment_simple(cv_image)
        if self.method == "owl":
            masked_img = self.segment_owl(cv_image)
        if self.method == "vit":
            masked_img = self.segment_vit(cv_image)
        
        return masked_img

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

    # nano sam segmentation of a chosen area/point
    def segment_simple(self, cv_image):
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
    
    # segment an object based on text promt (with open vocabulary object detector) - very slow 
    def segment_owl(self, cv_image):
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

    # segment an image with ADE20k classes 
    def segment_vit(self, cv_image):
        image = cv_image

        h, w = image.shape[:2]      # Realsense: 720x1280 (1.777) | th=512 tw=928 (1.812) || 512x704 (1.375) 
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

        image_tensor = self.transform(res_image).cuda()
        image_tensor = image_tensor.unsqueeze(0)

        # Run the network
        output = self.predictor(image_tensor)

        # pixel_collumn = output[0, :, 32, 44].cpu().numpy()

        print(output.shape)

        probs, max_output = torch.max(output, dim=1)
        pred = max_output[0].cpu().numpy()
        probs = probs[0].cpu().numpy()
        # pred = self.process_segmentation_output(output, 8, 0)

        canvas = self.get_canvas(image, pred, probs, CLASS_COLORS, no_overlay=False)

        return canvas
    
    # TODO: Segnet from jetson_inference 
    def segment_segnet(self, pil_image):
        pass

    def process_segmentation_output(self, probabilities, window_class, background_class):
        """
        Process the output from a semantic segmentation neural network to find pixels where a certain class
        has the maximum probability and take the second most probable class instead.

        Args:
            probabilities (torch.Tensor): A tensor of shape (num_pixels, num_classes) containing class probabilities for each pixel.
            window_class (int): The class index representing the window class.
            background_class (int): The class index representing the background class.

        Returns:
            torch.Tensor: A tensor of shape (num_pixels,) containing the processed class predictions for each pixel.
        """
        
        # Get the class indices with the maximum probability for each pixel
        max_class_indices = torch.argmax(probabilities, dim=1)
        
        # Create a mask for pixels where the window class has the maximum probability
        window_mask = max_class_indices == window_class

        # Get the top 2 class probabilities and indices for each pixel
        top2_probs, top2_indices = torch.topk(probabilities, k=3, dim=1)

        # Select the second most probable class for pixels where the window class has the maximum probability
        second_class_indices = top2_indices[window_mask, 1]

        # Create the final class predictions
        class_predictions = max_class_indices.clone()
        class_predictions[window_mask] = second_class_indices

        # Set the class predictions to the background class for pixels where the window class has the maximum probability
        # and the second most probable class is also the window class
        # background_mask = (window_mask) & (second_class_indices == window_class)
        # class_predictions[background_mask] = background_class

        return class_predictions[0].cpu().numpy()


    # Function to draw the segmentation mask on top of an image
    def get_canvas(self,
        image: np.ndarray,
        mask: np.ndarray,
        probs: np.ndarray,
        colors: tuple or list,
        opacity=0.7,
        no_overlay=False,
    ) -> np.ndarray:
        image_shape = image.shape[:2]
        mask_shape = mask.shape
        if image_shape != mask_shape:
            mask = cv2.resize(mask, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        seg_mask = np.zeros_like(image, dtype=np.uint8)

        if no_overlay:
            canvas = np.asarray(mask, dtype=np.uint8)
            return canvas

        frame_classes = []
        for k, color in enumerate(colors):
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
                canvas = cv2.putText(canvas, CLASSES[c], (10,30+45*i), cv2.FONT_HERSHEY_SIMPLEX, 1.8, colors[c], 2, cv2.LINE_AA)
            # Put the probability score for every prediction pixel on the image
            # for x in range(mask_shape[1]):
            #     for y in range(mask_shape[0]):
            #         # Get the coordintes of the mask pixel in image coordinates
            #         pix_size = int(image_shape[1]/mask_shape[1])+1
            #         img_x = pix_size*x
            #         img_y = pix_size*y+10
            #         canvas = cv2.putText(canvas, f"{-probs[y, x]:.1f}", (img_x, img_y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1, cv2.LINE_AA, False)

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
                "output"  # 
            ]
        )

        return mask_predictor_trt

if __name__ == "__main__":

    node = SAMNode(method="vit", 
                    debug=True,
                    synchronize=True
                    )
    img = cv2.imread("/home/appuser/data/shared_data/catkin_ws/src/sam_node/src/msg171507727-227585.jpg")
    mask = node.segment(img)

    cv2.imshow("mask", mask)
    cv2.waitKey()

    
