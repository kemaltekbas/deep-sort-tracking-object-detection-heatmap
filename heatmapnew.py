import argparse
import time
from pathlib import Path
import cv2
#if you do not have any error with OMP do not use it
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
parser = argparse.ArgumentParser()
opt = parser.parse_args()

classes_to_filter=[0]

import time

def xyxy_to_xywh(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_bottom = max([xyxy[1].item(), xyxy[3].item()])
    bbox_height = abs(xyxy[1].item() - xyxy[3].item())
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = bbox_height / 4
    
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_bottom - bbox_height / 4)
    w = bbox_w 
    h = bbox_h
    
    return x_c, y_c, w, h
        
def load_classes(path):
    # Loads *.names file at 'path'
    with open("data/coco.names", 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))

def video_detection(save_img= False):
    names, source, weights, view_img, save_txt, imgsz, trace = opt.names, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace 
    source = str(source)  # convert source to a string
    save_img = not opt.nosave and not source.endswith('.txt')

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
     ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
  
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                      max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                      nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                      max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                      use_cuda=True)
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'gpu'  # half precision only supported on CUDA
    
    #vcap = cv2.VideoCapture("rtsp://admin:Istech2021@192.168.1.188")
    vid_cap = cv2.VideoCapture(0)
    # Determine dimensions of video
    width = int(vid_cap.get(3))
    height = int(vid_cap.get(4))
    global global_img_np_array

    super_imposed_img = None
    global_img_np_array = np.ones([height, width], dtype = np.uint8)
    img_np_array = np.ones([height, width], dtype = int)

  # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
      model = TracedModel(model, device, opt.img_size)

    if half:
      model.half()  # to FP16

  # Second-stage classifier
    classify = False
    if classify:
      modelc = load_classifier(name='resnet101', n=0)  # initialize
      modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # total_detections = 0
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  
        dataset = LoadStreams(source, img_size=640)
    else:
        
        dataset = LoadImages(source, img_size=640)

    names = load_classes(names)
    
    if device.type != 'gpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]
                
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()
        
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
   

        # Iterate over the final detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique(): 
                    person_class_index = 0 
                    if int(c) == person_class_index: 
                        n = (det[:, -1] == c).sum()  
                        xywh_bboxs = []
                        confs = [] 
                        oids = []
                        
                        
                for *xyxy, conf, cls in reversed(det):
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                    oids.append(int(cls))
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    
                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)

                # Tracker inference
                outputs = deepsort.update(xywhs, confss,oids, im0)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id= outputs[:, -1]

                    # Extract tracked object's bounding box coordinates
                    for i, box in enumerate(bbox_xyxy):
                        x1, y1, x2, y2 = [int(i) for i in box]
                        print("global image np array",global_img_np_array)
                        # Increment frequency counter for whole bounding box
                        global_img_np_array[y1:y2, x1:x2] += 2
                        print("global image np array",global_img_np_array)

                    # Heatmap array preprocessing
                    if global_img_np_array.size != 0:
                       global_img_np_array_norm = (global_img_np_array - global_img_np_array.min()) / (global_img_np_array.max() - global_img_np_array.min()) * 255
                    else:
                       global_img_np_array_norm = global_img_np_array

                    global_img_np_array_norm = global_img_np_array_norm.astype('uint8')
                    print("global image np array norm",global_img_np_array_norm)

                    # Apply Gaussian blur and draw heatmap
                    global_img_np_array_norm = cv2.GaussianBlur(global_img_np_array_norm,(9,9), 0)
                    heatmap_img = cv2.applyColorMap(global_img_np_array_norm, cv2.COLORMAP_JET)

                    # Overlay heatmap on video frames
                    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, im0, 0.5, 0)
                    cv2.imshow('Heatmap', super_imposed_img)
                    vid_writer.release()  # release previous video writer

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, super_imposed_img)
                    print(f" The image with the result is saved in: {save_path}")
                   
                  
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                            
                           
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            global_img_np_array= np.ones([int(h),int(w)], dtype=np.uint32)
  
                        else:  # stream
                            fps, w, h = 120, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer.write(super_imposed_img)
        key = cv2.waitKey(1)  # Wait for 5 seconds
        if key == ord('q'):
           vid_cap.release()
           cv2.destroyAllWindows() 
                       
           


class Args:
    def __init__(self):
        self.cfg = 'deep_sort.yaml'
        self.names = 'coco.names'
        self.weights = 'yolov7-tiny.pt'
        #self.source = 'rtsp://admin:Istech2021@192.168.1.188'  # camera or video file path or image directory path
        self.source = "mall.mp4"
        self.output = 'inference/output'
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = 'gpu'
        self.view_img = True
        self.save_txt = False
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.update = False
        self.exist_ok = False
        self.project = 'runs/detect'
        self.name = 'exp'
        self.nosave = False
        self.no_trace = False
        self.trailslen= False
        self.classes = classes_to_filter

opt = Args()

video_detection(save_img= True)