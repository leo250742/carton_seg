# carton_seg
Carton segmentation model based on yolov8_seg improvement
# Getting Started
---
> This code is from the "Optimized space measurement for In-container palletizing robots: a vision-based approach" paper and is currently being submitted to The Visual computer journal
### Installation
  Dependencies:
- Pytorch+CUDA
https://pytorch.org/get-started/previous-versions/
- ZED SDK with Python API
- OpenCV
- YOLOv8
```
 python -m pip install ultraytics
```
- YOLOv10
```
git clone https://github.com/ultralytics/yolov10.git](https://github.com/leo250742/carton_seg.git
cd yolov10
pip install -r requirements.txt
```
### Data preparation
Because our data is protected by privacy, it cannot be made public, but we can say something about our data structure; You can reproduce with your own data set according to the following structure:

The data format is YOLO format. You can annotate the data with LabelImg software and convert it to YOLO format. The final data set is structured as follows, where img is used to store image data sets, txt is used to store image datasets in YOLO format
```
|- eld_person
    |- img
    |- txt
```
### Data preprocessing

You can divide the data set according to` Split_data.py` . Structure of the partitioned data set as
```
|- eld_person
    |- imges
        |-train
        |-test
        |-vain
    |- txt
        |-train
        |-test
        |-vain
```
You can enhance your data set according to `dataaugument.py`; Note that to prevent overfitting, you should only do data enhancement on train image data



### Mouth recognition

Training prepared data sets with YOLOv5, YOLOv8, YOLOv10



### Mouth location

* Once the ZED API is installed, start the ZED camera according to `get_python_api.py` and test the installation according to `hello_zed.py`

* `split_picture.py` is used to split the binocular camera picture, showing only the right (left) camera picture

* Connect the ZED camera and run `zedyolo_text.py` to realize real-time identification and positioning of the mouth, display 3D coordinates and depth distance.

* `keshihua_conf_data.py` is used to verify the detection of ZED cameras under different thresholds.

