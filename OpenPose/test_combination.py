import argparse
import os
import cv2
import numpy as np
from image_manager import ImageManager

parser = argparse.ArgumentParser(description='Run image manager')
set_name = "/SET2"
directory_path =  "../Images/Experiment" + set_name

# file_name_cam1 = directory_path + "/cam1_1103.jpg"
# file_name_cam2 = directory_path + "/cam2_1103.jpg"
file_name_cam1 = directory_path + "/cam1_120.jpg"
file_name_cam2 = directory_path + "/cam2_120.jpg"

parser.add_argument("--cam1", default = file_name_cam1, help="Input image_1")
parser.add_argument("--cam2", default = file_name_cam2, help="Input image_2")
args = parser.parse_args()


def merge(img_left, img_right):
    cv2.imshow('cam1', img_left)
    cv2.imshow('cam2', img_right)

     # stereo_image = np.concatenate((cam_1_image, cam_2_image), axis=1)

    width, height = img_left.shape[1], img_left.shape[0]

    # Define the distance between the cameras and the focal length
    d = 0
    f = 4.22
    # Calculate the horizontal shift for the two images
    shift = d * f / (2 * f - d)
    print (shift)

    # Define the transformation matrix for the left image
    M_left = np.array([[1, 0, 0], [0, 1, 0], [shift, 0, 1]])

    # Define the transformation matrix for the right image
    M_right = np.array([[1, 0, 0], [0, 1, 0], [-shift, 0, 1]])

    # Apply the perspective transformation to the left image
    img_left_transformed = cv2.warpPerspective(img_left, M_left, (width, height))

    # Apply the perspective transformation to the right image
    img_right_transformed = cv2.warpPerspective(img_right, M_right, (width, height))

    # Combine the two images
    merged = cv2.addWeighted(img_left_transformed, 0.5, img_right_transformed, 0.5, 0)

    # Display the merged image
    cv2.imshow('Merged', merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def avg_image(img_left, img_right):
    image_data = []
    image_data.append(img_left)
    image_data.append(img_right)

    avg_image = image_data[0]
    for i in range(len(image_data)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(image_data[i], alpha, avg_image, beta, 0.0)

    # cv2.imwrite('avg_happy_face.png', avg_image)
    # avg_image = cv2.imread('avg_happy_face.png')
    
    cv2.imshow('avg', avg_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__== "__main__":    
    img_left = cv2.imread(args.cam1)
    img_right = cv2.imread(args.cam2)
    
    # Method 1
    # merge(img_left, img_right)

    # Method 2
    avg_image(img_left, img_right)
