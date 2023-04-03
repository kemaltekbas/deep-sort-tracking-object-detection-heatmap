import argparse
from deep_sort_tracking_id import xyxy_to_tlwh,xyxy_to_xywh,compute_color_for_labels,draw_boxes,UI_box,draw_border
from utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import torch
from utils.torch_utils import select_device,time_synchronized
import time
import torch.backends.cudnn as cudnn
from utils.datasets import LoadStreams,LoadImages
import cv2
import numpy as np
from utils.general import non_max_suppression, scale_coords,strip_optimizer
from deep_sort_tracking_id import xyxy_to_xywh
classes_to_filter = [0]
from collections import deque
from numpy import random
import os
import torch

def select_device(device=''):
    device = 'cpu'
    return torch.device(device)



data_deque = {} 
opt  = {
    
    "weights": "yolov7-tiny.pt", # Path to weights file default weights are for nano model
    "yaml"   : "data/coco.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.25, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : '0',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter,  # list of classes to filter or None
    "config_deepsort" : "deep_sort_pytorch/configs/deep_sort.yaml",
    "augment": False,
    "agnostic-nms": False

}

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    print(f"img shape: {shape}")
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def video_detection(path_x='' ,conf_=0.25):
  source = path_x
  webcam = path_x == '0' or path_x.startswith(
        'rtsp') or path_x.startswith('http') or path_x.endswith('.txt')
  cfg = get_config()
  cfg.merge_from_file(opt["config_deepsort"])
  
  deepsort = DeepSort("deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7",
                      max_dist=0.2, min_confidence=0.1,
                      nms_max_overlap=0.5, max_iou_distance=0.7,
                      max_age=70, n_init=3, nn_budget=100,
                      use_cuda=True)
  torch.cuda.empty_cache()
  # Initializing model and setting it for inference
  with torch.no_grad():
    device = select_device(opt["device"])
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = torch.load(opt["weights"], map_location=device)[
        'model'].float()
    model.to(device).eval()
    if half:
        model.half()
    

    start_time = time.time()
    # total_detections = 0
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  
        dataset = LoadStreams(source, img_size=640)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=640)

    names = model.module.names if hasattr(model, 'module') else model.names

    t0 = time.time()
    img = torch.zeros((1, 3, 640, 640), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    vcap = cv2.VideoCapture(path_x)

    # Determine dimensions of video
    width = int(vcap.get(3))
    height = int(vcap.get(4))

    global global_img_np_array

    super_imposed_img = None
    global_img_np_array = np.ones([height, width], dtype = np.uint32)
    img_np_array = np.ones([height, width], dtype = int)



    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        torch.cuda.empty_cache()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Detector inference
        t1 = time_synchronized()
        pred = model(img, augment=opt["augment"])[0]

        # Perform NMS
        pred = non_max_suppression(
            pred, conf_, opt["iou-thres"], classes=opt["classes"], agnostic=opt["agnostic-nms"])
        t2 = time_synchronized()

        # Iterate over the final detections
        for i, det in enumerate(pred):
            
            if webcam:
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]
            # save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    detection_ = int(n)
                    s += '%g %ss, ' % (n, names[int(c)])

                bbox_xywh = []
                confs = []

                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Tracker inference
                outputs = deepsort.update(xywhs, confss, im0)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)

                    # Extract tracked object's bounding box coordinates
                    for i, box in enumerate(bbox_xyxy):
                        x1, y1, x2, y2 = [int(i) for i in box]
                        

                        # Increment frequency counter for whole bounding box
                        global_img_np_array[y1:y2, x1:x2] += 1

                        img_np_array = cv2.rotate(img_np_array, cv2.ROTATE_180)
                        img_np_array = cv2.flip(img_np_array, 1)
                        

                    # Heatmap array preprocessing
                    global_img_np_array_norm = (global_img_np_array - global_img_np_array.min()) / (global_img_np_array.max() - global_img_np_array.min()) * 255
                    global_img_np_array_norm = global_img_np_array_norm.astype('uint8')

                   # Apply Gaussian blur and draw heatmap
                    global_img_np_array_norm = cv2.GaussianBlur(global_img_np_array_norm,(9,9), 0)
                    heatmap_img = cv2.applyColorMap(global_img_np_array_norm, cv2.COLORMAP_JET)

                   # Overlay heatmap on video frames
                    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, im0, 0.5, 0)
                    
                    # Display the resulting frame
                    #cv2.imshow('object detection', img)
                    # Press any key to exit
                    #if cv2.waitKey(0) == ord('q'):
                       #break


                   # Visualize heatmap 
                    yield super_imposed_img,detection_
"""                    
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='path to input video file')
    args = parser.parse_args()
    video_detection(args.video)

"""                    
                   # 
                   
               

"""
detectedObjects = 0
def generate_frames(path_x = '',conf_= 0.25):
    yolo_output = video_detection(path_x,conf_)
    
    for detection_,detection in yolo_output:
        # # print(detection_)
        ref,buffer=cv2.imencode('.jpg',detection_)
       

        global detectedObjects
        detectedObjects = str(detection)
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')  
 
"""









