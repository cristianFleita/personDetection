import os
import cv2
import json
from image_manager import ImageManager
from src.openpose import OpenPoseDetector
import config

protoFile = "pose/body_25/pose_deploy.prototxt"
weightsFile = "pose/body_25/pose_iter_584000.caffemodel"

set_name = config.met_test_set
directory_path = config.directory_path_met_test


if __name__ == "__main__":
    image_manager = ImageManager()
    images = image_manager.load_images_from_directory(directory_path)

    pose_detector = OpenPoseDetector(protoFile, weightsFile)
    matrix_detection = []
    detections = {
        "file_name": "",
        "image_keypoints": ""
    }

    for image in images:
        file_name = image.get("file_name")
        print(file_name)

        (detection_frame, image_key_points) = pose_detector.detect_key_points(image.get("image"))
        cv2.imwrite("results/Images" + set_name + "/" + file_name, detection_frame)
        detections["file_name"] = file_name
        detections["image_keypoints"] = image_key_points
        matrix_detection.append(detections.copy())

    json_object = json.dumps(matrix_detection, indent=4)
    folder_name = os.path.basename(directory_path).split('/')[-1]

    with open('results/persons_skeletons_keypoints_' + folder_name + '.json', 'w+') as json_file:
        json_file.write(json_object)
