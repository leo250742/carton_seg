from ultralytics import YOLO
model = YOLO('消融实验/train_seg_n/weights/best.pt')
#model.load('/home/bns/lihao/yolov11n.pt')# 加载预训练模型，如果本地没有会自动下载
model.train(data="/home/bns/lh/pywork/datasets/OSCD/data.yaml", epochs=10, batch=32, imgsz=640,device="0")
# metrics=model.val(data="/home/bns/lh/pywork/datasets/OSCD/data.yaml",save_json=True)
# print(f"mAP:{metrics.seg.map}")
# print(f"mAP50:{metrics.seg.map50}")
# print(f"mAP75:{metrics.seg.map75}")
# # from ultralytics import YOLO
# # model = YOLO('/home/bns/lihao/ultralytics/cfg/models/v8/yolov8-pose.yaml')
# # model.load('/home/bns/lihao/yolov8n.pt')# 加载预训练模型，如果本地没有会自动下载
# # model.train(data="/home/bns/sevenT/xhy/sea/mmpose_dataset/yolodataset/c.yaml", epochs=300, batch=32, imgsz=640,device="0")
