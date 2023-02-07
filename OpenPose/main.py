import argparse
import os
import cv2
import json
from image_manager import ImageManager
from openpose import process_keypoints

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument("--rectify_params", default = "params_py_sensor_1.xml", help="Input params_py_sensor_1")
args = parser.parse_args()

set_name = "/SET3"
directory_path =  "../Images/Experiment" + set_name

# set_name = "/TEST"
# directory_path =  "../Images" + set_name

rectify_folder = "../Images/Rectify" + set_name

def rectify_image(image, file_name, out_name):
    camID = file_name[0:4]
    cv_file = cv2.FileStorage(args.rectify_params, cv2.FILE_STORAGE_READ)
            
    Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
    Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
    Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
    Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
    cv_file.release()

    # ImgL = Cam1 , ImgR = Cam2
    if (camID == "cam1"):
        image= cv2.remap(image,Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    elif (camID == "cam2"):
        image= cv2.remap(image,Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    cv2.imwrite(out_name, image)

def apply_rectification(image_manager, directory_path):
    images = image_manager.load_images_from_directory(directory_path)
    for image in images:
        file_name = image.get("file_name")
        file = image.get("image")
        out_name = rectify_folder + "/" + file_name
        rectify_image(file, file_name, out_name)
        
if __name__== "__main__":
    image_manager = ImageManager()
    # apply_rectification(image_manager, directory_path)

    images = image_manager.load_images_from_directory(directory_path)
    
    matrix_detection = []
    detections = { 
                    "file_name": "",
                    "image_keypoints" : ""
                    }
    
    for image in images:
        file_name = image.get("file_name")
        print(file_name)
        image_keypoints = process_keypoints(image.get("image"), "Results/Images" + set_name + "/" + file_name)
        
        detections["file_name"] = file_name
        detections["image_keypoints"] = image_keypoints
        matrix_detection.append(detections.copy())
    
    json_object = json.dumps( matrix_detection, indent = 4 )
    folder_name = os.path.basename(directory_path).split('/')[-1]

    with open('Results/persons_skeletons_keypoints_'+ folder_name + '.json', 'w+') as json_file:
        json_file.write(json_object)