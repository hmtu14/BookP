from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
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

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    print("------------------")
    print("In predict tranform")
    print("Prediction shape: {}".format(prediction.shape))
    print("Inp dimesion: {}".format(inp_dim))

    batch_size = prediction.size(0)
    #Stride is the length of each grid cell (e.g. image size 416x416, divide into 13x13 cells so stride = 416/13 = 32(pixel))
    stride =  inp_dim // prediction.size(2)
    print("Stride: {}".format(stride))
    grid_size = inp_dim // stride
    print("Grid size: {}".format(grid_size))
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    print("Number of anchors: {}".format(num_anchors))
    
    print("------Begin prediction transformation:")
    print("Prediction shape: {}".format(prediction.shape))

    #change dimension from B*(A*85)*S*S to B*(A*85)*(S*S):
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    print("Prediction shape: {}".format(prediction.shape))
    #change dimension from B*(A*85)*(S*S) to B*(S*S)*(A*85):
    prediction = prediction.transpose(1,2).contiguous()
    print("Prediction shape: {}".format(prediction.shape))
    #change dimension from B*(A*S*S)*85:
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    print("Prediction shape: {}".format(prediction.shape))

    print("Anchor: ")
    print(anchors)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    print("New anchor: ")
    print(anchors)
    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
    print("Prediction final shape: {}".format(prediction.shape))
    print("------------------")
    return prediction

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    print("------------------")
    print("In write results")
    print("Prediction shape: {}".format(prediction.shape))
    #conf_mask: if bounding box's confidence is larger than confidence, conf_mask equal 1, else 0
    #conf_mask dimension: 1x10647x1 (10647 = (52x52 + 26x26 + 13x13)x3)
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    print("conf_mask shape: {}".format(conf_mask.shape))
    prediction = prediction*conf_mask
    
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    #Box corner shape dimension: 1x10647x85
    print("box_corner shape: {}".format(box_corner.shape))
    
    batch_size = prediction.size(0)

    write = False
    


    for ind in range(batch_size):
        print("*********************")
        image_pred = prediction[ind]          #image Tensor
        #confidence threshholding 
        #NMS
        print(image_pred[:,5:5+ num_classes].shape)
        #For each bounding box, get the class with highest confidence:
        #from shape 10647x80, get the max class of each bounding box (max of 80) to get 10647x1 shape
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        print(max_conf)
        print("max_conf_score shape: {}".format(max_conf_score.shape))
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        print("image_pred[:,:5] shape: {}".format(image_pred[:,:5].shape))
        print("max_conf shape: {}".format(max_conf.shape))
        print("max_conf_score shape: {}".format(max_conf_score.shape))
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue
#        
  
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index
        print("img_classes shape: {}".format(img_classes.shape))
        
        for cls in img_classes:
            #perform NMS

        
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections
            
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
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
    
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names
