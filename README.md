# carton_seg
Carton segmentation model based on yolov8_seg improvement
# Getting Started
---
> This code is from the "Optimized space measurement for In-container palletizing robots: a vision-based approach" paper and is currently being submitted to The Visual computer journal
### Installation
  Dependencies:
- Pytorch+CUDA
https://pytorch.org/get-started/previous-versions/
- OpenCV
- YOLOv8
```
 python -m pip install ultraytics
```
### Data preparation
Because our data is protected by privacy, it cannot be made public, but we can say something about our data structure; You can reproduce with your own data set according to the following structure:

The data format is YOLO format. You can annotate the data with LabelImg software and convert it to YOLO format. The final data set is structured as follows, where img is used to store image data sets, txt is used to store image datasets in YOLO format
```
|- carton
    |- img
    |- txt
```
### Data preprocessing

You can divide the data . Structure of the partitioned data set as
```
|- carton
    |- imges
        |-train
        |-test
        |-vain
    |- txt
        |-train
        |-test
        |-vain
```
### Train
Once the dataset is prepared, run the train.py file in the main directory to start training, and select the hyperparameters that are suitable for your device.
