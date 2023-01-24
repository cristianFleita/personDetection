import cv2
import os

class ImageManager:
    def __init__(self, folder_path:str):
        self.folder_path = folder_path

    def __init__(self):
        self.folder_path = ""

    def load_image(self, file_name):
        return cv2.imread(file_name)

    def load_images_from_folder(self, folder_path):
        images = []
        
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path,filename))
            if img is not None:
                images.append(img)
        
        return images

    def load_images_from_folder(self):
        return self.load_images_from_folder(self.folder_path)

    def save_image(self, image, file_name):
        cv2.imwrite(file_name, image)

    def show_image(self, image):
        # Display the image 
        cv2.imshow("Image", image) 

        # Display the window until any key is pressed
        cv2.waitKey(0) 
        # Close all windows
        cv2.destroyAllWindows()