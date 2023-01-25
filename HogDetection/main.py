from hog_detection import HogDetector
from image_manager import ImageManager

IMAGE_PATH = "../Images/pedestrians_1.jpg"

if __name__== "__main__":
    detector = HogDetector()
    image_manager = ImageManager()

    images = image_manager.load_images_from_folder("../Images")
    matrix_detection = []
    detections = { 
                    "file_name": "",
                    "bounding_boxes" : ""
                    }

    for image in images:
        image_bounding_boxes = detector.generate_bounding_boxes(image.get("image"))
        detections["file_name"] = image.get("file_name")
        detections["bounding_boxes"] = image_bounding_boxes
        matrix_detection.append(detections.copy())
    
    print(matrix_detection)