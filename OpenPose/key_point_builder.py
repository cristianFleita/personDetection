from config import KeyPointConfig
from key_point_detector import OpenPoseMultiPersonDetector

class KeyPointBuilder:
    
    def __init__(self, model_id):
        if (model_id == "body_25"):
            self.model = self.create_model_25()
            self.detector = self.create_detector()
    
    def create_model_25(self) -> KeyPointConfig: 
        return KeyPointConfig("body_25")

    def create_detector(self):
        return OpenPoseMultiPersonDetector(self.model)

    def get_detector(self):
        return self.detector