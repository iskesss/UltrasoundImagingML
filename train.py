from roboflow import Roboflow
from ultralytics import YOLO
import torch

def run():
    torch.multiprocessing.freeze_support()
    
    # Initialize Roboflow
    rf = Roboflow(api_key="PonCX4D1tGssuHy2wkxd")
    project = rf.workspace("roses").project("fetal-hc")
    version = project.version(1)
    dataset = version.download("yolov11")

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
    # Load a COCO-pretrained YOLO11n model
    model = YOLO("yolo11n.pt").to(device)

    # Train the model on the COCO8 example dataset for 30 epochs
    results = model.train(data="Fetal-HC-1/data.yaml", epochs=10, imgsz=640)

    # Uncomment this line to run inference on an image
    # results = model("path/to/bus.jpg")

if __name__ == '__main__':
    run()
