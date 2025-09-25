from ultralytics import YOLO
model = YOLO("best.pt")
result = model(source="images/test",conf=0.8,save=True,show_labels=False,show_conf=False,name="output")
#result = model(source="/home/bns/lh/pywork/datasets/OSCD/images/test",save=True,save_conf=True,save_txt=True,name="output")
print(model.info())
