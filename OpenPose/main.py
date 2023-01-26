import argparse
import os
import cv2
from image_manager import ImageManager

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on")
args = parser.parse_args()

directory_path =  "../Images/Experiment/SET2"

if __name__== "__main__":
    image_manager = ImageManager()
    directory_images = image_manager.load_images_from_directory(directory_path)
    print(directory_images)