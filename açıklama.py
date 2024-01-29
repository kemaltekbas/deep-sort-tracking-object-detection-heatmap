import argparse
import time
from pathlib import Path
import cv2
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
parser.add_argument('--save_conf', action='store_true', help='çıktı metin dosyasına güvenceleri kaydedin')
opt = parser.parse_args()


classes_to_filter=[0]

"""
xyxy[0]: Bounding box'ın sol üst köşesinin x koordinatı.
xyxy[1]: Bounding box'ın sol üst köşesinin y koordinatı.
xyxy[2]: Bounding box'ın sağ alt köşesinin x koordinatı.
xyxy[3]: Bounding box'ın sağ alt köşesinin y koordinatı.
"""

def xyxy_to_xywh(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = max([xyxy[1].item(), xyxy[3].item()])
    bbox_height = abs(xyxy[1].item() - xyxy[3].item())
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = bbox_height / 4
    
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top - bbox_height / 8)
    w = bbox_w 
    h = bbox_h
    
    return x_c, y_c, w, h

def a(*xyxy):
    
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h
"""
bu fonksiyon, belirtilen yoldaki bir .names 
dosyasındaki tüm sınıf isimlerini bir liste olarak döndürür.
"""      
def load_classes(path):    
    with open("data/coco.names", 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))

def video_detection(save_img= False):
    # opt adlı nesneden bazı değerleri alıp yerel değişkenlere atar. 
    names, source, weights, view_img, save_txt, imgsz, trace = opt.names, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace 
    # source değişkenini bir stringe dönüştürür.
    source = str(source)  
    save_img = not opt.nosave and not source.endswith('.txt')

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
     ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    #get_config() fonksiyonuyla bir yapılandırma nesnesi alır ve 
    #deep_sort_pytorch/configs/deep_sort.yaml dosyasından ayarları bu nesneye ekler.
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    
    #DeepSort'un parametreleri, daha önce ayarlanan cfg_deep değişkeninden alınan değerlerle belirlenir 
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                      max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                      nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                      max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                      use_cuda=True)
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'gpu'  # half precision only supported on CUDA
    
    #vid_cap = cv2.VideoCapture("rtsp://admin:Istech2021@192.168.1.188")
    vid_cap = cv2.VideoCapture("mall.mp4")
    # Determine dimensions of video
    width = int(vid_cap.get(3))
    height = int(vid_cap.get(4))
    """
3 veya cv2.CAP_PROP_FRAME_WIDTH: Videonun genişliği.
4 veya cv2.CAP_PROP_FRAME_HEIGHT: Videonun yüksekliği.
5 veya cv2.CAP_PROP_FPS: Videonun kare hızı (FPS).
    """     
    global global_img_np_array
    super_imposed_img = None
    global_img_np_array = np.ones([height, width], dtype = np.uint8)
    img_np_array = np.ones([height, width], dtype = int)

    #Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    #Bu model, FP32 (tek hassasiyetli kayan nokta) formatında yüklenir.
    stride = int(model.stride.max())  # model stride
    """
    Modelin adımlama değeri, genellikle konvolüsyonel sinir 
    ağlarında bir konvolüsyon işlemi sırasında kaç piksel 
    atlanacağını belirtir. Bu değer, modelin mimarisine bağlı olarak belirlenir.
    """
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
      model = TracedModel(model, device, opt.img_size)

    if half:
      model.half()  # to FP16
    
    #Second-stage classifier
    classify = False
    
    if classify:
      modelc = load_classifier(name='resnet101', n=0)  # initialize
      modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    
    
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  
        dataset = LoadStreams(source, img_size=640)
    else:    
        dataset = LoadImages(source, img_size=640)

    names = load_classes(names)
    
    #if device.type != 'gpu':
        #model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    t0 = time.time()
    #dataset içerisindeki her görüntü için bir döngü başlatılır. Her frameden bahsediyor
    for path, img, im0s, vid_cap in dataset:
        #Görüntü, numpy dizisinden PyTorch tensoruna dönüştürülür.
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
        #derin öğrenme modeli ile gerçek zamanlı çıkarım (inference) yapma ve sonrasında elde edilen tahminleri işleme adımlarını içerir
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()
        """
        Özetle, bu kod parçacığı, derin öğrenme modeliyle gerçek zamanlı çıkarım yapma,
        bu çıkarımın ne kadar sürdüğünü ölçme ve sonrasında tahminleri NMS ile işleme adımlarını 
        gerçekleştirir.
        """
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
            #pred:model tarafından üretilen tahminleri içeren tensor ya da liste.

        # Iterate over the final detections
        """
        pred tensor olup her bir görüntü için nesne algılamalarını içerir.
        enumerate fonksiyonu, her algılama için bir indeks (i) ve 
        algılamaların kendisini (det) döndürür.
        """
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            """
            Bu kısım, gerekli dosya yollarını oluşturur. save_path görüntünün kaydedileceği yolu,
            txt_path ise metin bilgilerinin (eğer varsa) kaydedileceği yolu belirtir.
            """
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            output_folder = "outputs"
            txt_path = str(Path(output_folder) / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}') + '.txt'

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            #Eğer det listesi/tesirinde eleman varsa (yani görüntüde bir şey tespit edildiyse) kodun geri kalanını çalıştır.
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique(): 
                    person_class_index = 0 
                    if int(c) == person_class_index: 
                        n = (det[:, -1] == c).sum()  
                        #Bu, mevcut görüntüde tespit edilen toplam kişi sayısını hesaplar.
                        """
                        Bu üç liste, sonraki adımlarda kullanılmak üzere boş
                        olarak oluşturulur. xywh_bboxs sınırlayıcı kutuları, 
                        confs güven değerlerini ve oids nesne ID'lerini saklamak 
                        için kullanılır.
                        """
                        xywh_bboxs = []
                        confs = [] 
                        oids = []
                        
                        
                """
                Bu kod parçacığı, her bir tespiti işleyerek sınırlayıcı
                kutu koordinatlarını ve diğer ilgili bilgileri toplar ve
                bu bilgileri metin dosyasına kaydeder.
                """
                for *xyxy, conf, cls in reversed(det):
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                    oids.append(int(cls))
                    """
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
"""
                    if save_txt:
                        # Convert xyxy list to a tensor
                        xyxy_tensor = torch.tensor(xyxy)
                        
                        # Dönüşümü yap
                        xywh = xyxy_to_xywh(xyxy_tensor)
                        cls, x_c, y_c, w, h, conf = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format

                        # Extract scalar values from tensors
                        cls = cls.item()
                        x_c = x_c.item()
                        y_c = y_c.item()
                        w = w.item()
                        h = h.item()

                        # line değişkenini başlat
                        line = [cls, x_c, y_c, w, h, conf] if opt.save_conf else [cls, x_c, y_c, w, h]

                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % tuple(line) + '\n')



                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)

                # Tracker inference
                outputs = deepsort.update(xywhs, confss,oids, im0)
                
                """
                Bu satır, deepsort adında bir izleme (tracking) algoritması olan DeepSORT'un güncellenmesini sağlar.
                DeepSORT, nesneleri video kareleri arasında izlemek için kullanılan bir algoritmadır.

                xywhs: Nesne sınırlayıcı kutularının koordinatları.
                confss: Nesne algılamalarının güven skorları.
                oids: Nesne algılamalarının sınıf kimlikleri (IDs).
                im0: İşlenen görüntü.
                deepsort.update() fonksiyonu, bu girdi bilgilerini kullanarak izlemenin güncellenmesini sağlar ve 
                izlenen nesnelerin güncellenmiş koordinatlarına dair bilgileri döndürür.

                Sonuç olarak, bu kod parçası, nesne algılamalarını ve bu algılamaların güven skorlarını alır ve bu bilgileri
                kullanarak nesnelerin izlenmesini sağlar. Bu sayede, video kareleri arasında nesnelerin hareketini izleyebilirsiniz.
               """
               
                if len(outputs) > 0:
                    """
                    Bu kontrol, outputs değişkeninin boş olup olmadığını kontrol eder. 
                    outputs, DeepSORT algoritmasının sonucunda elde edilen izlenen nesnelerin 
                    bilgilerini içerir. Eğer outputs boş değilse, aşağıdaki kod bloğu çalıştırılır.
                    """
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id= outputs[:, -1]

                    # Extract tracked object's bounding box coordinates
                    for i, box in enumerate(bbox_xyxy):
                        #Bu döngü, izlenen her bir nesnenin sınırlayıcı kutusunu tek tek işler.
                        x1, y1, x2, y2 = [int(i) for i in box]
                        print("global image np array",global_img_np_array)
                        # Increment frequency counter for whole bounding box
                        global_img_np_array[y1:y2, x1:x2] += 2
                        print("global image np array",global_img_np_array)
                        """
                        DeepSORT algoritmasından elde edilen izlenen nesnelerin
                        sınırlayıcı kutularını işler ve bu sınırlayıcı kutuların
                        yer aldığı global_img_np_array değişkenindeki piksel değerlerini artırır. 
                        Bu, izlenen nesnelerin belirli bir bölgede bulunduğunu işaretlemek için yapılır
                        """

                    # Heatmap array preprocessing
                    if global_img_np_array.size != 0:
                       global_img_np_array_norm = (global_img_np_array - global_img_np_array.min()) / (global_img_np_array.max() - global_img_np_array.min()) * 255
                    # Normalize etme, bir dizi veya görüntünün değerlerini belirli bir aralığa (bu durumda 0-255) ölçeklendirme işlemidir. Bu, görüntü işleme işlemleri için gereklidir.
                    else:
                       global_img_np_array_norm = global_img_np_array

                    global_img_np_array_norm = global_img_np_array_norm.astype('uint8')
                    print("global image np array norm",global_img_np_array_norm)

                    # Apply Gaussian blur and draw heatmap
                    global_img_np_array_norm = cv2.GaussianBlur(global_img_np_array_norm,(9,9), 0)
                    #Bu satır, normalize edilmiş diziye Gaussian bulanıklık uygular. Bu, yoğunluk haritasındaki gürültüyü azaltmak ve görüntüyü yumuşatmak için kullanılır.
                    heatmap_img = cv2.applyColorMap(global_img_np_array_norm, cv2.COLORMAP_JET)
                    #Bu satır, normalize edilmiş ve bulanıklaştırılmış diziyi renkli bir yoğunluk haritasına dönüştürür. Burada 'JET' renk haritası kullanılır, bu tipik bir renk paletidir.
                    
                    super_imposed_img = cv2.addWeighted(heatmap_img, 0.4, im0, 0.6, 0)
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
        self.conf_thres = 0.4
        self.iou_thres = 0.6
        self.device = 'gpu'
        self.view_img = True
        self.save_txt = True
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
        self.save_conf= True

opt = Args()

video_detection(save_img= True)