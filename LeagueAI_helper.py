#Copyright 2019 Oliver Struckmeier
#Licensed under the GNU General Public License, version 3.0. See LICENSE for details

from mss import mss
from PIL import Image
import cv2
import numpy as np

class detection:
    def __init__(self, object_class, x_min, y_min, x_max, y_max, confidence):
        self.object_class = object_class
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.w = abs(x_max - x_min)
        self.h = abs(y_max - y_min)
        self.x = x_min + int(self.w/2)
        self.y = y_min + int(self.h/2)
        self.confidence = confidence
    def toString(self):
        print("class: {}, confidence: {}, min: ({}|{}), max: ({}|{}), width: {}, height: {}, center: ({}|{})".format(self.object_class, self.confidence, self.x_min, self.y_min, self.x_max, self.y_max, self.w, self.h, self.x, self.y))

class input_output:
    def __init__(self, input_mode, SCREEN_WIDTH=None, SCREEN_HEIGHT=None, video_filename=None):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.input_mode = input_mode
        self.video_filename = video_filename
        if input_mode == 'webcam':
            self.capture_device = cv2.VideoCapture(0)
            assert(self.capture_device.isOpened()), 'Error could not open capture device for Webcam -1'
        elif input_mode == 'videofile':
            assert(self.video_filename is not None), "Error please enter a valid video file name"
            self.capture_device = cv2.VideoCapture(self.video_filename)
            assert self.capture_device.isOpened(), 'Error could not open capture device for Videofile: {}'.format(self.video_filename)
        elif input_mode == 'desktop':
            assert(SCREEN_HEIGHT is not None and SCREEN_HEIGHT is not None), "Error please set SCREEN_WIDTH and SCREEN_HEIGHT"
            self.capture_device = mss()
            self.mon = {'top': 0, 'left': 0, 'width' : self.SCREEN_WIDTH, 'height' : self.SCREEN_HEIGHT}
        else:
            raise Exception('Unknown input mode!')

    def get_pixels(self, output_size=None):
        if self.input_mode == 'webcam':
            ret, frame = self.capture_device.read()
            assert(ret == True), 'Error: could not retrieve frame'
            return frame
        if self.input_mode == 'videofile':
            ret, frame = self.capture_device.read()
            assert(ret == True), 'Error: could not retrieve frame'
            return frame
        elif self.input_mode == 'desktop':
            frame = self.capture_device.grab(self.mon)
            screen = Image.frombytes('RGB', frame.size, frame.bgra, "raw", "BGRX")
            # Swap R and B channel
            R, G, B = screen.split()
            screen = Image.merge("RGB", [B, G, R])
            screen = np.array(screen)
            if output_size == None:
                output_size = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            screen = cv2.resize(screen, output_size)
            return screen
        else:
            raise Exception('Unknown input mode!')

import torch
import torch.nn as nn
from torch.autograd import Variable
from yolov3_detector import Detector
import pickle as pkl
import random

class LeagueAIFramework():
    def __init__(self, config_file, weights, names_file, classes_number, resolution, threshold, cuda=True, nms_confidence=0.4, draw_boxes=True):
        self.config_file = config_file
        self.weights = weights
        self.names_file = names_file
        self.names = self.load_classes(self.names_file)
        self.classes = classes_number
        self.resolution = resolution
        assert (self.resolution > 0), "Error: Resolution out of bounds"
        self.threshold = threshold
        assert (self.threshold >= 0 or self.threshold > 1.0), "Error, threshold out of bounds!"
        # Create a new detector object with the given config file
        self.detector = Detector(self.config_file)
        print("Created new detector with config: ", self.config_file)
        # Load the weights and set up the network
        self.detector.load_weights(self.weights)
        self.detector.net_info["height"] = self.resolution
        # Make sure the resolution is a multiple of 32
        self.input_dimension = int(self.detector.net_info["height"])
        assert (self.input_dimension % 32 == 0 and self.input_dimension > 32), "Error the resolution is not a multiple of 32!"
        self.cuda = cuda
        self.nms_conf = nms_confidence
        self.draw_boxes = draw_boxes
        print("Network ", weights, " loaded succesfully!")
        if self.cuda:
            if torch.cuda.is_available():
                self.detector.cuda()
                print("Model running on GPU")
            else:
                self.cuda = False
                print("Cuda not available, running on CPU")
        else:
            print("Model running on CPU")
    def get_objects(self, input_frame):
        # Preprocess the data to put into the neural network
        im_dim = input_frame.shape[1], input_frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)
        frame = self.preprocess_frame(input_frame)
        if self.cuda:
            im_dim = im_dim.cuda()
            frame = frame.cuda()
        # Run the detection
        with torch.no_grad():
            output = self.detector(Variable(frame), self.cuda)
        output = self.write_results(output)

        if type(output) == int or output.size(0) <= 0:
            print("No detection")
            return []

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(int(self.resolution)/im_dim,1)[0].view(-1,1)
                
        output[:,[1,3]] -= (self.input_dimension - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (self.input_dimension - scaling_factor*im_dim[:,1].view(-1,1))/2
                
        output[:,1:5] /= scaling_factor
                
        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
        if self.draw_boxes:
            list(map(lambda x: self.draw_results(x, input_frame), output))

        # Create a dict of all objects with their x,y center pos and width/height
        detected_objects = []
        for i in range(output.shape[0]):
            d = detection(output[i][7], output[i][1], output[i][2], output[i][3], output[i][4], output[i][5])
            detected_objects.append(d)
        return detected_objects

    def draw_results(self, x, results):
        font_size = 1
        font_thickness = 2
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results
        cls = int(x[-1])
        color = (255,0,0)
        label = "{0}".format(self.names[cls])
        cv2.rectangle(img, c1, c2,color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, font_size, [225,255,255], font_thickness);
        return img

    def write_results(self, prediction):
        conf_mask = (prediction[:,:,4] > self.threshold).float().unsqueeze(2)      
        prediction = prediction*conf_mask
            
        box_corner = prediction.new(prediction.shape)
        box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
        box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
        box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
        box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
        prediction[:,:,:4] = box_corner[:,:,:4]

        write = False

        image_pred = prediction[0]          #image Tensor
        # Threshholding 
        # NMS
     
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ self.classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            print("Error")
        
        #Get the various classes detected in the image
        img_classes = self.unique(image_pred_[:,-1])  # -1 index holds the class index

        for cls in range(0, self.classes):
            # Perform NMS
            # Get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            # Sort the detections such that the entry with the maximum objectness
            # Confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections
            
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at in the loop
                try:
                    ious = self.bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
                #Get the IOUs of all boxes that come after the one we are looking at in the loop
                try:
                    ious = self.bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
             
                except IndexError:
                    break
             
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < self.nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
             
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(0)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class
             
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out)) 
        try:
            return output
        except:
            return 0

    def bbox_iou(self, box1, box2):
        """
        This function computes the Intersection or Union of two bounding boxes
        """
        #Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
        #get the corrdinates of the intersection rectangle
        inter_rect_x1 =  torch.max(b1_x1, b2_x1)
        inter_rect_y1 =  torch.max(b1_y1, b2_y1)
        inter_rect_x2 =  torch.min(b1_x2, b2_x2)
        inter_rect_y2 =  torch.min(b1_y2, b2_y2)
            
        #Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
            
        #Union Area
        b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
            
        iou = inter_area / (b1_area + b2_area - inter_area)
            
        return iou

    def unique(self, tensor):
        tensor_np = tensor.cpu().numpy()
        unique_np = np.unique(tensor_np)
        unique_tensor = torch.from_numpy(unique_np)
         
        tensor_res = tensor.new(unique_tensor.shape)
        tensor_res.copy_(unique_tensor)
        return tensor_res

    def preprocess_frame(self, frame):
        """
        This function preprocesses the given frame to input to the neural network
        """
        # Resize the frame
        w, h = self.input_dimension, self.input_dimension
        frame_w, frame_h = frame.shape[1], frame.shape[0]
        new_w = int(frame_w * min(w / frame_w, h / frame_h)) 
        new_h = int(frame_h * min(w / frame_w, h / frame_h))
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
        # Padding depending on the aspect ratio
        padding = np.full((self.input_dimension, self.input_dimension, 3), 128)
        padding [(h-new_h)//2:(h-new_h)//2 + new_h, (w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_frame
        # Done with resizing and padding
        frame = padding
        # transpose the frame array into RGB channels, width, height
        frame = frame[:, :,::-1].transpose((2,0,1)).copy()
        # Make tensor and noralise to 1 (divide by 255)
        frame = torch.from_numpy(frame).float().div(255.0).unsqueeze(0)
        return frame
    
    def load_classes(self, names_file):
        f = open(names_file, "r")
        names = f.read().split("\n")[:-1]
        return names
        
        
        


