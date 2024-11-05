from roboflow import Roboflow
from ultralytics import YOLO
import torch
import os
from PIL import Image

def train():
    torch.multiprocessing.freeze_support()

    #Import dataset from Roboflow
    rf = Roboflow(api_key="PonCX4D1tGssuHy2wkxd")
    project = rf.workspace("mhf-model").project("training-set-gqfju-wv7jx")
    version = project.version(2)
    dataset = version.download("yolov11")
                
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    #Print CUDA information if available
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
    results = model([os.path.join("training-set-2/test/images", filename) for filename in os.listdir("training-set-2/test/images")])
        
    # Directory paths
    test_images_dir = "training-set-2/test/images"
    
    # Create a new results file if it already exists, increment folder number to distinguish new tests
    result_images_dir = "result-images-0"
    counter = 1
    # Incrementing to update folder name by version
    while(os.path.exists(result_images_dir)):
        result_images_dir = result_images_dir[:len(result_images_dir) - 1] + str(counter)
        counter += 1
    
    # Make a new folder directory
    os.makedirs(result_images_dir)
        
    # Add all test images into array in image_filenames
    image_filenames = [filename for filename in os.listdir(test_images_dir) if filename.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Run test images through model to get predictions based on pretrained model weights
    results = model([os.path.join(test_images_dir, filename) for filename in image_filenames])
    
    # Process results list
    for i, result in enumerate(results):
        #boxes = result.boxes  # Boxes object for bounding box outputs
        #masks = result.masks  # Masks object for segmentation masks outputs
        #keypoints = result.keypoints  # Keypoints object for pose outputs
        #probs = result.probs  # Probs object for classification outputs
        #obb = result.obb  # Oriented boxes object for OBB outputs

        # Save result image to new folder
        result_filename = os.path.join(result_images_dir, f"result_{image_filenames[i]}")
        result.save(filename=result_filename)  # save to disk

        # Compare and combine test image with result image side by side
        test_image_path = os.path.join(test_images_dir, image_filenames[i])
        combined_image = combine_images_side_by_side(test_image_path, result_filename)

        # Save the combined image
        combined_image.save(os.path.join(result_images_dir, f"combined_{image_filenames[i]}"))

# Helper function to visualize predictions and compare side-by-side with test images
def combine_images_side_by_side(test_image_path, result_image_path):
    # Open the test and result images
    test_image = Image.open(test_image_path)
    result_image = Image.open(result_image_path)

    # Ensure both images have the same height by resizing the result image
    if test_image.size[1] != result_image.size[1]:
        result_image = result_image.resize((int(result_image.size[0] * test_image.size[1] / result_image.size[1]), test_image.size[1]))

    # Create a new image with combined width and same height as the test image
    combined_width = test_image.width + result_image.width
    combined_image = Image.new('RGB', (combined_width, test_image.height))

    # Paste the test image on the left and the result image on the right
    combined_image.paste(test_image, (0, 0))
    combined_image.paste(result_image, (test_image.width, 0))

    return combined_image

# Main
if __name__ == '__main__':
    #train()
    test()
