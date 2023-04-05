# deep-sort-tracking-object-detection-heatmap
A full directory of implementation of YOLOv7 and a basic heatmap construction.

- Go to the folder that you downloaded.
cd path to/deep-sort-tracking-object-detection-heatmap

- Install the dependecies
pip install -r requirements.txt

The main parts of this project is "deep_sort_tracking_id.py,detect.py,heatmap.py"

- This is the link that provide "yolov7-tiny.pt" [weights file]:https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
you should add this to your folder.

- This the the link that provide "deep_sort_pytorch" :https://drive.google.com/drive/folders/1kna8eWGrSfzaR6DtNJ8_GchGgPMv3VC8
you should unzip it and add into main folder.

- The "deep_sort_tracking.py" visualiation starts from the terminal. 
cd path to/deep-sort-tracking-object-detection-heatmap

python deep_sort_tracking_id.py --weights yolov7-tiny.pt  --img-size 640  --source 0 (for you should change the name of .pt file) 
python deep_sort_tracking_id.py --weights yolov7-tiny.pt  --img-size 640 --source "mall.mp4" --view-img(for getting a faster visualization you can change the size)

- The "detect.py" visualiation starts from the terminal. .
cd path to/deep-sort-tracking-object-detection-heatmap
python detect.py --weights yolov7-tiny.pt  --img 640  --source 0
python detect.py --weights yolov7-tiny.pt  --img-size 640 --source "mall.mp4" --view-img

- The "heatmap.py" visualiation starts from the terminal.
cd path to/deep-sort-tracking-object-detection-heatmap
python heatmap.py --weights yolov7-tiny.pt  --source "mall.mp4" --view-img

 I will be glad if you want to ask questions in any part or give a tip about the project.

![tracking](https://user-images.githubusercontent.com/127952905/229531174-5fd796be-83cd-4dae-9d30-f472c7afdd8f.jpg)

![detection](https://user-images.githubusercontent.com/127952905/229532478-74d2437c-1a19-4449-92d5-f03e4174da01.jpg)

![heatmap](https://user-images.githubusercontent.com/127952905/230157715-aa765dda-eeae-41a2-b9b0-665b9e89ca59.jpg)

CREDIT

https://github.com/MuhammadMoinFaisal
https://github.com/DoganK01
