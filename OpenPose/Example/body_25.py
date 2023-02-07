import cv2
import time
import math
import numpy as np
from config import *


class general_mulitpose_model(object):
    def __init__(self, keypoint_num):
        self.point_names = point_name_25 if keypoint_num == 25 else point_names_18
        self.point_pairs = point_pairs_25 if keypoint_num == 25 else point_pairs_18
        self.map_idx = map_idx_25 if keypoint_num == 25 else map_idx_18
        self.colors = colors_25 if keypoint_num == 25 else colors_18
        self.num_points = 25 if keypoint_num == 25 else 18

        self.prototxt = prototxt_25 if keypoint_num == 25 else prototxt_18
        self.caffemodel = caffemodel_25 if keypoint_num == 25 else caffemodel_18
        self.pose_net = self.get_model()

    def get_model(self):
        coco_net = cv2.dnn.readNetFromCaffe(self.prototxt, self.caffemodel)
        return coco_net

    def predict(self, imgfile):
        start = time.time()
        img = cv2.imread(imgfile)
        height, width, _ = img.shape
        net_height = 368
        net_width = int((net_height / height) * width)
        start = time.time()

        in_blob = cv2.dnn.blobFromImage(
            img, 1.0 / 255, (net_width, net_height), (0, 0, 0), swapRB=False, crop=False)
        self.pose_net.setInput(in_blob)
        output = self.pose_net.forward()
        print("[INFO]Time Taken in Forward pass: {} ".format(time.time() - start))
        detected_keypoints = []
        keypoints_list = np.zeros((0, 3))
        keypoint_id = 0
        threshold = 0.1
        for part in range(self.num_points):
            probMap = output[0, part, :, :]
            probMap = cv2.resize(probMap, (width, height))

            keypoints = self.getKeypoints(probMap, threshold)
            print("Keypoints - {} : {}".format(self.point_names[part], keypoints))
            keypoint_with_id = []
            for i in range(len(keypoints)):
                keypoint_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1
            detected_keypoints.append(keypoint_with_id)
        valid_paris, invalid_pairs = self.getValidPairs(output, detected_keypoints, width, height)
        personwiseKeypoints = self.getPersonwiseKeypoints(valid_paris, invalid_pairs, keypoints_list)
        img = self.vis_pose(imgfile, personwiseKeypoints, keypoints_list)
        FPS = math.ceil(1 / (time.time() - start))
        img = cv2.putText(img, "FPS" + str(int(FPS)), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        return img

    def getKeypoints(self, probMap, threshold=0.1):
        mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
        mapMask = np.uint8(mapSmooth > threshold)
        keypoints = []

        # find the blobs
        contours, _= cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))
        return keypoints

    def getValidPairs(self, output, detected_keypoints, width, height):
        valid_pairs = []
        invalid_pairs = []
        n_interp_samples = 15
        paf_score_th = 0.1
        conf_th = 0.7

        for k in range(len(self.map_idx)):
            # A -> B constitute a limb
            pafA = output[0, self.map_idx[k][0], :, :]
            pafB = output[0, self.map_idx[k][1], :, :]
            pafA = cv2.resize(pafA, (width, height))
            pafB = cv2.resize(pafB, (width, height))

            candA = detected_keypoints[self.point_pairs[k][0]]
            candB = detected_keypoints[self.point_pairs[k][1]]
            nA = len(candA)
            nB = len(candB)
            if (nA != 0 and nB != 0):
                valid_pair = np.zeros((0, 3))
                for i in range(nA):
                    max_j = -1
                    maxScore = -1
                    found = 0
                    for j in range(nB):
                        # Find d_ij
                        d_ij = np.subtract(candB[j][:2], candA[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        # Find p(u)
                        interp_coord = list(
                            zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                        # Find L(p(u))
                        paf_interp = []
                        for k in range(len(interp_coord)):
                            paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                               pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
                        # Find E
                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores) / len(paf_scores)
                        # check if the connection is valid
                        # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                        if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                            if avg_paf_score > maxScore:
                                max_j = j
                                maxScore = avg_paf_score
                                found = 1
                    # Append the connection to the list
                    if found:
                        valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

                # Append the detected connections to the global list
                valid_pairs.append(valid_pair)

            else:  # If no keypoints are detected
                print("No Connection : k = {}".format(k))
                invalid_pairs.append(k)
                valid_pairs.append([])

        return valid_pairs, invalid_pairs

    def getPersonwiseKeypoints(self, valid_pairs, invalid_pairs, keypoints_list):
        personwiseKeypoints = -1 * np.ones((0, self.num_points + 1))
        for k in range(len(self.map_idx)):
            if k not in invalid_pairs:
                partAs = valid_pairs[k][:, 0]
                partBs = valid_pairs[k][:, 1]
                indexA, indexB = np.array(self.point_pairs[k])
                for i in range(len(valid_pairs[k])):
                    found = 0
                    person_idx = -1
                    for j in range(len(personwiseKeypoints)):
                        if personwiseKeypoints[j][indexA] == partAs[i]:
                            person_idx = j
                            found = 1
                            break
                    if found:
                        personwiseKeypoints[person_idx][indexB] = partBs[i]
                        personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + \
                                                               valid_pairs[k][i][2]
                    elif not found and k < self.num_points - 1:
                        row = -1 * np.ones(self.num_points + 1)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) + \
                                  valid_pairs[k][i][2]
                        personwiseKeypoints = np.vstack([personwiseKeypoints, row])
        return personwiseKeypoints

    def vis_pose(self, img_file, personwiseKeypoints, keypoints_list):
        img = cv2.imread(img_file)
        print("Detected:",len(personwiseKeypoints))
        for i in range(self.num_points - 1):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(self.point_pairs[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(img, (B[0], A[0]), (B[1], A[1]), self.colors[i], 3, cv2.LINE_AA)
        img = cv2.resize(img, (480, 640))
        return img

if __name__ == '__main__':
    gmm = general_mulitpose_model(25)
    img = gmm.predict("images/pedestrians_2.jpg")
    cv2.imshow("frame", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()