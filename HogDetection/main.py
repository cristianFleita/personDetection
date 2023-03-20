from hog_detection import HogDetector
from image_manager import ImageManager
import json
import os
import cv2

# DIRECTORY_PATH = "../Images/Experiment/SET3"
# DIRECTORY_PATH = "../Images/Test"

DIRECTORY_PATH = "Images\Test"

def show_bounding_boxes(image_manager, image, image_bounding_boxes):
    for bounding_boxes in image_bounding_boxes:
        x = bounding_boxes["x"]
        y = bounding_boxes["y"]
        w = bounding_boxes["width"]
        h = bounding_boxes["height"]

        cv2.rectangle(image.get("image"), 
                        (x, y),  
                        (x + w, y + h),  
                        (0, 0, 255), 
                        4)
        image_manager.show_image(image.get("image"))

if __name__== "__main__":
    #Read the folder in windows
    file_dir = os.path.split (os.path.dirname(__file__)) [0]
    file_name = os.path.join(file_dir, DIRECTORY_PATH)
    print (file_name)

    detector = HogDetector()
    image_manager = ImageManager()

    images = image_manager.load_images_from_folder(file_name)
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

        # show_bounding_boxes(image_manager, image, image_bounding_boxes)
    
    json_object = json.dumps( matrix_detection, indent = 4 )
    folder_name = os.path.basename(DIRECTORY_PATH).split('/')[-1]

    with open('results/bounding_boxes_'+ folder_name + '.json', 'w+') as json_file:
        json_file.write(json_object)