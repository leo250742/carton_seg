from ultralytics import YOLO
model = YOLO('/ultralytics/cfg/models/v8/SK+neck.yaml'')
model.load('yolov8n.pt')# 加载预训练模型，如果本地没有会自动下载
model.train(data="data.yaml", epochs=300, batch=32, imgsz=640,device="0")


