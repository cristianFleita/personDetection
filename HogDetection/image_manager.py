import cv2
import os
import numpy as np

class ImageManager:
    def __init__(self):
        self.folder_path = ""

    def load_image(self, file_name):
        return cv2.imread(file_name)

    #https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
    def load_images_from_folder(self, folder_path):
        images =  []
        dict = {
                "file_name": "",
                "image": ""
                }
        
        for file_name in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path,file_name))
            if img is not None:
                name = os.path.basename(file_name).split('/')[-1]
                dict["file_name"] = name
                dict["image"] = img
                images.append(dict.copy())
        
        return images

    def save_image(self, image, file_name):
        cv2.imwrite(file_name, image)

    def show_image(self, image):
        # Display the image 
        cv2.imshow("Image", image) 

        # Display the window until any key is pressed
        cv2.waitKey(0) 
        # Close all windows
        cv2.destroyAllWindows()