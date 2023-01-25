from hog_detection import HogDetector
from image_manager import ImageManager
import json
import os

DIRECTORY_PATH = "../Images/Experiment/SET3"

if __name__== "__main__":
    detector = HogDetector()
    image_manager = ImageManager()

    images = image_manager.load_images_from_folder(DIRECTORY_PATH)
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
    
    json_object = json.dumps( matrix_detection, indent = 4 )
    folder_name = os.path.basename(DIRECTORY_PATH).split('/')[-1]

    with open('Results/bounding_boxes_'+ folder_name + '.json', 'w+') as json_file:
        json_file.write(json_object)
