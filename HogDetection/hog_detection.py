import numpy as np
import cv2

class HogDetector():
    
    def __init__(self):
        # Create a HOGDescriptor object
        self.hog = cv2.HOGDescriptor()
        # Initialize the People Detector
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def generate_bounding_boxes(self, image):
        (bounding_boxes, weights) = self.hog.detectMultiScale(image, winStride=(4, 4),
                                        padding=(8, 8), 
                                        scale=1.05)
        bounding_boxes_array = []
        dict = {
                "x": "",
                "y": "",
                "width":"",
                "height":"",
                }
        for (x, y, w, h) in bounding_boxes:
            dict["x"] = int(x)
            dict["y"] = int(y)
            dict["width"] = int(w)
            dict["height"] = int(h)
            bounding_boxes_array.append(dict.copy())
            
        return bounding_boxes_array

    