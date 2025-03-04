# Install the ultralytics library if not already installed
# !pip install ultralytics

# Import necessary libraries
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

KMP_DUPLICATE_LIB_OK="TRUE"

# Load the trained model
model_path = 'D:/TIH_Projects/COEOGE/Code/yolo/best.pt'
model = YOLO(model_path)

# Function to perform inference on a single image
def predict_image(image_path, conf_threshold=0.55):
    
    # Run prediction
    """
    Perform inference on a single image using the trained model.

    Args:
        image_path (str): Path to the image file
        conf_threshold (float, optional): Confidence threshold (default: 0.55)

    Returns:
        None

    Prints the class labels, confidence scores, and bounding boxes of the detected objects
    Visualizes the results by drawing bounding boxes on the input image
    """
    results = model.predict(source=image_path, conf=conf_threshold)

    # Extract results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class labels

        # Print results
        for box, score, cls in zip(boxes, scores, classes):
            print(f"Class: {cls}, Score: {score}, Box: {box}")

        # Visualize results
        img = cv2.imread(image_path)
        for box in boxes:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

# Example usage
image_path = 'D:/TIH_Projects/COEOGE/Code/yolo/IMG_0007_2.tif'  # Replace with your image path
predict_image(image_path)
