from roboflow import Roboflow
from ultralytics import YOLO

rf = Roboflow(api_key="PonCX4D1tGssuHy2wkxd")
project = rf.workspace("roses").project("fetal-hc")
version = project.version(1)
dataset = version.download("yolov11")


# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="Fetal-HC-1/data.yaml", epochs=30, imgsz=640, device="mps")

# Run inference with the YOLO11n model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")
