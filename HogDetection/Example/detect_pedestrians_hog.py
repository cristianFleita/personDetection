import cv2  # Import the OpenCV library to enable computer vision
import json
import numpy as np

# Description: Detect pedestrians in an image using the
#   Histogram of Oriented Gradients (HOG) method

filename = "../../Images/Experiment/SET3/cam1_1193.jpg"

def main():
    # Create a HOGDescriptor object
    hog = cv2.HOGDescriptor()

    # Initialize the People Detector
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Load an image
    image = cv2.imread(filename)

    # Detect people
    # image: Source image
    # winStride: step size in x and y direction of the sliding window
    # padding: no. of pixels in x and y direction for padding of sliding window
    # scale: Detection window size increase coefficient   
    # bounding_boxes: Location of detected people
    # weights: Weight scores of detected people
    (bounding_boxes, weights) = hog.detectMultiScale(image, 
                                                    winStride=(4, 4),
                                                    padding=(8, 8), 
                                                    scale=1.05)

    # Draw bounding boxes on the image
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(image, 
                        (x, y),  
                        (x + w, y + h),  
                        (0, 0, 255), 
                        4)

    print(type(bounding_boxes))    
    bounding_boxes_list =bounding_boxes.tolist()

    json_object = json.dumps(bounding_boxes_list, indent=4)

    with open('bounding_boxes.json', 'w') as json_file:
        json_file.write(json_object)

    # Create the output file name by removing the '.jpg' part
    size = len(filename)
    new_filename = filename[:size - 4]
    new_filename = new_filename + '_detect.jpg'

    # Save the new image in the working directory
    cv2.imwrite(new_filename, image)

    # Display the image 
    cv2.imshow("Image", image) 

    # Display the window until any key is pressed
    cv2.waitKey(0) 
    # Close all windows
    cv2.destroyAllWindows() 

main()