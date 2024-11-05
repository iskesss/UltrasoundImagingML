from roboflow import Roboflow
from ultralytics import YOLO
import torch
import os

def run():
    torch.multiprocessing.freeze_support()
    
    # Initialize Roboflow
    #rf = Roboflow(api_key="PonCX4D1tGssuHy2wkxd")
    #project = rf.workspace("roses").project("fetal-hc")
    #version = project.version(1)
    #dataset = version.download("yolov11")


    #rf = Roboflow(api_key="PonCX4D1tGssuHy2wkxd")
    #project = rf.workspace("mhf-model").project("training-set-gqfju-wv7jx")
    #version = project.version(1)
    #dataset = version.download("yolov11")
    

    rf = Roboflow(api_key="PonCX4D1tGssuHy2wkxd")
    project = rf.workspace("mhf-model").project("training-set-gqfju-wv7jx")
    version = project.version(2)
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

    # Train the model on the COCO8 example dataset for 10 epochs
    results = model.train(data="training-set-2/data.yaml", epochs=10, imgsz=640)

    # Uncomment this line to run inference on an image
    # results = model("path/to/bus.jpg")
    
def test():
    # Load a model
    model = YOLO("./runs/detect/train27/weights/best.pt")  # pretrained YOLO11n model

    # Run batched inference on a list of images
    #for filename in os.listdir("training-set-2"):
    #    print(filename)
    #results = model([filename for filename in os.listdir("training-set-2/test/images")])  # return a list of Results objects
    
    results = model(['training-set-2/test/images/546_HC_png.rf.e3957fda110520171af9e832bd820fb7.jpg'])
    #results = model([file for file in os.listdir('training-set-2')])
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk
    
if __name__ == '__main__':
    #run()
    test()
