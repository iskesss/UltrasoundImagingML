from roboflow import Roboflow
from ultralytics import YOLO
import torch
import os
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import supervision as sv
import urllib.request
import matplotlib.pyplot as plt

from ultralytics.data.annotator import auto_annotate

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


def crop_to_bbox(image_path, box):
    """Crop the image to the bounding box region."""
    # Convert tensor to numpy array if necessary
    if isinstance(box, torch.Tensor):
        box = box.cpu().numpy()  # Convert tensor to numpy array

    # Ensure box is a 1D array (i.e., single bounding box)
    if len(box.shape) > 1:  
        box = box[0]  # If it's a 2D array, take the first box (or loop through if needed)

    # Extract box coordinates (x_min, y_min, x_max, y_max)
    x_min, y_min, x_max, y_max = map(int, box)

    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Crop the image using the bounding box coordinates
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Convert back to PIL Image for saving
    return Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    
def test():
    # Load a model
    model = YOLO("./runs/detect/train27/weights/best.pt")  # pretrained YOLO11n model
        
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
    
    # Create a new bboxes file if it already exists, increment folder number to distinguish new tests
    bbox_images_dir = "bboxes-0"
    counter = 1
    # Incrementing to update folder name by version
    while os.path.exists(bbox_images_dir):
        bbox_images_dir = bbox_images_dir[:-1] + str(counter)
        counter += 1
    
    # Make a new folder directory
    os.makedirs(bbox_images_dir)
    
        
    # Add all test images into array in image_filenames
    image_filenames = [filename for filename in os.listdir(test_images_dir)]
    
    # Run test images through model to get predictions based on pretrained model weights
    results = model([os.path.join(test_images_dir, filename) for filename in image_filenames])
    
    # Process results list
    for i, result in enumerate(results):
        boxes = result.boxes.xyxy  # Boxes object for bounding box outputs

        # Save result image to new folder
        result_filename = os.path.join(result_images_dir, f"result_{image_filenames[i]}")
        result.save(filename=result_filename)  # save to disk

        # Compare and combine test image with result image side by side
        test_image_path = os.path.join(test_images_dir, image_filenames[i])
        combined_image = combine_images_side_by_side(test_image_path, result_filename)

        # Save the combined image
        combined_image.save(os.path.join(result_images_dir, f"combined_{image_filenames[i]}"))
        
        # Crop image to bounding box and save to bbox folder
        bbox_image = crop_to_bbox(test_image_path, boxes)
        bbox_image.save(os.path.join(bbox_images_dir, f"bbox_{image_filenames[i]}"))


def download_checkpoint(CHECKPOINT_PATH):
    # Download if the file does not exist
    if not os.path.isfile(CHECKPOINT_PATH):
        print("Checkpoint not found. Downloading...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        urllib.request.urlretrieve(url, CHECKPOINT_PATH)
        print("Download complete.")

def segment():
    HOME = os.getcwd()
    CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
    
    # Ensure the weights directory exists
    os.makedirs(os.path.join(HOME, "weights"), exist_ok=True)
    
    # Download the checkpoint if it doesn't exist
    download_checkpoint(CHECKPOINT_PATH)
    print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))
    
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
    sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    # Create a new segmented file if it already exists, increment folder number to distinguish new tests
    segmented_dir = "segmented-0"
    counter = 1
    # Incrementing to update folder name by version
    while os.path.exists(segmented_dir):
        segmented_dir = segmented_dir[:-1] + str(counter)
        counter += 1
    
    # Make a new folder directory
    os.makedirs(segmented_dir)
    
    # Test the segmentation
    for filename in os.listdir("bboxes-2"):
        file_path = os.path.join("bboxes-2", filename)
        
        image_bgr = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sam_result = mask_generator.generate(image_rgb)

        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        detections = sv.Detections.from_sam(sam_result=sam_result)
        annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        
        output_path = os.path.join(segmented_dir, f"segmented_{filename}")
        cv2.imwrite(output_path, annotated_image)
        print(f"Segmented image saved at: {output_path}")
        
        combined_image = combine_images_side_by_side(file_path, output_path)
        combined_image.save(os.path.join(segmented_dir, f"combined_segmented_{filename}"))
        #sv.plot_images_grid(
        #images=[image_bgr, annotated_image],
        #grid_size=(1, 2),
        #titles=['source image', 'segmented image']
        #)
# Main
if __name__ == '__main__':
    #train()
    #test() 
    segment()