import argparse
import os
import cv2
import json
from image_manager import ImageManager
from openpose import process_keypoints

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on")
args = parser.parse_args()

directory_path =  "../Images/Experiment/SET3"

if __name__== "__main__":
    image_manager = ImageManager()
    images = image_manager.load_images_from_directory(directory_path)
    
    matrix_detection = []
    detections = { 
                    "file_name": "",
                    "image_keypoints" : ""
                    }
    
    for image in images:
        file_name = image.get("file_name")
        image_keypoints = process_keypoints(image.get("image"), "Results/Images/SET3/"+file_name)
        
        detections["file_name"] = file_name
        detections["image_keypoints"] = image_keypoints
        matrix_detection.append(detections.copy())
    
    # kps_array = process_keypoints(image, filename)
    
    json_object = json.dumps( matrix_detection, indent = 4 )
    folder_name = os.path.basename(directory_path).split('/')[-1]

    with open('Results/bounding_boxes_'+ folder_name + '.json', 'w+') as json_file:
        json_file.write(json_object)