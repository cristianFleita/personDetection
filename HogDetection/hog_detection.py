import numpy as np
import cv2

class HogDetector():
    
    def __init__(self):
        print("Not implemented")
        # Create a HOGDescriptor object
        self.hog = cv2.HOGDescriptor()
        # Initialize the People Detector
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def person_detection(self, image):
        self.hog.detectMultiScale(image, winStride=(4, 4),
                                        padding=(8, 8), 
                                        scale=1.05)