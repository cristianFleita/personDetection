import cv2
import os
import argparse


class ImageManager:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Run image manager')

    def load_image(self, file_name):
        file_name = self.parse_directory(file_name, self.parser)
        return cv2.imread(file_name), file_name

    def save_image(self, image, file_name):
        cv2.imwrite(file_name, image)

    def show_image(self, image):
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def parse_directory(self, directory, parser):
        parser.add_argument("--image_file", default=directory, help="Input image")
        args = parser.parse_args()

        return args.image_file

    def load_images_from_directory(self, directory_path):
        self.directory_path = self.parse_directory(directory_path, self.parser)

        images = []
        dict = {
            "file_name": "",
            "image": ""
        }

        for file_name in os.listdir(self.directory_path):
            img = cv2.imread(os.path.join(self.directory_path, file_name))
            if img is not None:
                name = os.path.basename(file_name).split('/')[-1]
                dict["file_name"] = name
                dict["image"] = img
                images.append(dict.copy())

        return images
