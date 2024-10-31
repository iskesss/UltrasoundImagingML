import cv2
import os


def fit_bounding_boxes() -> dict:
    """
    Detects bounding boxes around ellipses in annotated images within a specified directory and
    returns their coordinates in a format compatible with YOLO object detection models.

    This function iterates through all images in the "data/training_set" directory, specifically
    looking for images ending in "HC_Annotation.png". For each image:

    1. It loads the image in grayscale format.
    2. Applies a binary threshold to isolate the white elliptical region on a black background.
    3. Finds contours in the binary image, assuming one main ellipse per image.
    4. Calculates a bounding box around each detected contour.
    5. Normalizes the bounding box coordinates (center x, center y, width, height) relative to the image size.
    6. Stores the YOLO-compatible bounding box annotation as a dictionary entry with the filename as the key.
    7. Optionally, draws the bounding box on the image for visual confirmation and displays it briefly.

    Returns:
        dict: A dictionary where each key is the filename of an annotated image, and each value is a
              YOLO-formatted string containing the normalized bounding box annotation in the form:
              "class_id x_center y_center width height".

    Note:
        - This function requires OpenCV (`cv2`) for image processing.
        - The class ID is hardcoded as "0" since it assumes only one class (ellipse).
        - Each image window is displayed briefly and closed automatically; pressing a key will close the window immediately.
        - All OpenCV windows are closed once processing is complete.

    Example:
        bounding_boxes = fit_bounding_boxes()
        print(bounding_boxes)
    """
    boundingboxes = dict()

    for filename in os.listdir("data/training_set"):
        if filename.endswith("HC_Annotation.png"):

            print(f"Processing {filename}")

            # Load image
            image_path = os.path.join("data/training_set", filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Apply binary threshold to isolate the ellipse
            _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(
                binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Assume one main ellipse per image
            for contour in contours:
                # Get bounding box around the contour
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate center, width, and height for YOLO format (normalized)
                img_height, img_width = image.shape
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height

                # Append annotation as a single bounding box for an ellipse
                boundingboxes[filename[:6] + ".png"] = (x_center, y_center, width, height)

                # Optional: Draw bounding box on image for visualization
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display the image with the drawn rectangle
            #cv2.imshow("Image with Bounding Box", image)
            # Wait for a key press to close the window, else closes after one millisecond
            #cv2.waitKey(1)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    return boundingboxes


print(fit_bounding_boxes())
