import cv2
import time
import numpy as np
from config import KeyPointConfig

class OpenPoseMultiPersonDetector:
    def __init__(self, model: KeyPointConfig):
        self.model = model
        self.IN_HEIGHT = 368
        self.THRESHOLD = 0.1

    def run_person_detectection(self, image, filename):
        FRAME_WIDTH = image.shape[1]
        FRAME_HEIGHT = image.shape[0]
        output = self.load_net(image, self.model, FRAME_WIDTH, FRAME_HEIGHT)
        
        (keypoints_list, detected_keypoints) =  self.find_key_points(output, self.model,
                                                                    FRAME_WIDTH, FRAME_HEIGHT)

        
    
    def load_net(self, image, model: KeyPointConfig, FRAME_WIDTH, FRAME_HEIGHT ):
        t = time.time()
        net = cv2.dnn.readNetFromCaffe(model.get_proto_file(), model.get_weights_file())
        
        # Set up the network to work with cpu
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

        INPUT_WIDTH = int((self.IN_HEIGHT / FRAME_HEIGHT) * FRAME_WIDTH)
        INPUT_BLOB = cv2.dnn.blobFromImage(image, 1.0 / 255, (INPUT_WIDTH, self.IN_HEIGHT),
                            (0, 0, 0), swapRB = False, crop = False)
        net.setInput(INPUT_BLOB)

        return net.forward()

    def find_key_points(self, output, model: KeyPointConfig, FRAME_WIDTH, FRAME_HEIGHT):
        detected_keypoints = []
        # COORD1 = X,  COOR2= Y, COORD3 = CONFIDENSE
        keypoints_list = np.zeros((0,3))
        keypoint_id = 0

        for part in range(model.get_n_points()):
            probMap = output[0,part,:,:]
            probMap = cv2.resize(probMap, (FRAME_WIDTH, FRAME_HEIGHT))

            keypoints = self.get_key_points(probMap, self.THRESHOLD)
            
            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1
            
            detected_keypoints.append(keypoints_with_id)

        return keypoints_list , detected_keypoints

    def get_key_points(self, probMap, threshold = 0.1):
        mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
        mapMask = np.uint8(mapSmooth > threshold)
        keypoints = []

        #find the blobs
        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #for each blob find the maxima
        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

        return keypoints

    def find_person_skeleton(self):
        print("")