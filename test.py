from ultralytics import YOLO
model = YOLO("/home/bns/lihao/runs/segment/4/weights/best.pt")
result = model(source="/home/bns/lh/pywork/datasets/OSCD/images/test",conf=0.8,save=True,show_labels=False,show_conf=False,name="output")
#result = model(source="/home/bns/lh/pywork/datasets/OSCD/images/test",save=True,save_conf=True,save_txt=True,name="output")
print(model.info())