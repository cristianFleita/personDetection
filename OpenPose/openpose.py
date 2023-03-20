import cv2
import time
import numpy as np
import body25Config


class OpenPoseDetector:
    def __init__(self, proto_file, weights_file, device='cpu'):
        self.PROTO_FILE_PATH = proto_file
        self.WEIGHTS_FILE_PATH = weights_file
        self.DEVICE = device
        self.KEY_POINTS_NUMBER = body25Config.KEY_POINTS_NUMBER
        self.KEY_POINTS_MAPPING = body25Config.KEY_POINTS_MAPPING
        self.POSE_PAIRS = body25Config.POSE_PAIRS
        self.MAP_IDX = body25Config.MAP_IDX
        self.KEY_POINT_COLORS = body25Config.KEY_POINTS_COLORS

    def detect_key_points(self, image):
        frame_width = image.shape[1]
        frame_height = image.shape[0]

        init_time = time.time()
        net = self.load_net(image)
        output = net.forward()

        detected_key_points, key_points_list = self.find_all_key_points(output, image)

        (detection_frame, skeletons_detected) = self.detect_persons_skeletons(image, frame_width, frame_height, output,
                                                                              detected_key_points, key_points_list)

        print("Time Taken to detect key points in the image: {}".format(time.time() - init_time))
        return detection_frame, skeletons_detected

    def load_net(self, input_image):
        frame_width = input_image.shape[1]
        frame_height = input_image.shape[0]

        net = cv2.dnn.readNetFromCaffe(self.PROTO_FILE_PATH, self.WEIGHTS_FILE_PATH)
        self.set_net_back_end(net)

        # Fix the input Height and get the width according to the Aspect Ratio
        in_height = 368
        in_width = int((in_height / frame_height) * frame_width)

        inp_blob = cv2.dnn.blobFromImage(input_image, 1.0 / 255, (in_width, in_height),
                                         (0, 0, 0), swapRB=False, crop=False)

        net.setInput(inp_blob)
        return net

    def set_net_back_end(self, net):
        if self.DEVICE == "cpu":
            net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU device")
        elif self.DEVICE == "gpu":
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using GPU device")

    def find_all_key_points(self, output, image):
        frame_width = image.shape[1]
        frame_height = image.shape[0]

        detected_key_points = []
        key_points_list = np.zeros((0, 3))
        keypoint_id = 0
        threshold = 0.1

        for part in range(self.KEY_POINTS_NUMBER):
            prob_map = output[0, part, :, :]
            prob_map = cv2.resize(prob_map, (frame_width, frame_height))
            # print("ProbMap = {}".format(probMap))

            key_points = self.get_key_points(prob_map, threshold)

            # name_part = keypointsMapping[part]
            # print("{} : {}".format(keypointsMapping[part], keypoints))
            key_points_with_id = []
            for i in range(len(key_points)):
                key_points_with_id.append(key_points[i] + (keypoint_id,))
                key_points_list = np.vstack([key_points_list, key_points[i]])
                keypoint_id += 1

            detected_key_points.append(key_points_with_id)

        return detected_key_points, key_points_list

    def get_key_points(self, probability_map, threshold=0.1):
        map_smooth = cv2.GaussianBlur(probability_map, (3, 3), 0, 0)

        map_mask = np.uint8(map_smooth > threshold)
        key_points = []

        # find the blobs
        contours, _ = cv2.findContours(map_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # for each blob find the maxima
        for cnt in contours:
            blob_mask = np.zeros(map_mask.shape)
            blob_mask = cv2.fillConvexPoly(blob_mask, cnt, 1)
            masked_prob_map = map_smooth * blob_mask
            _, max_val, _, max_loc = cv2.minMaxLoc(masked_prob_map)
            key_points.append(max_loc + (probability_map[max_loc[1], max_loc[0]],))

        return key_points

    def detect_persons_skeletons(self, image, frame_width, frame_height, output, detected_key_points, key_points_list):
        frame_clone = image.copy()
        self.draw_key_points(frame_clone, detected_key_points)

        valid_pairs, invalid_pairs = self.get_valid_pairs(output, frame_width, frame_height,
                                                          detected_key_points)
        person_wise_key_points = self.get_person_wise_key_points(valid_pairs, invalid_pairs,
                                                                 key_points_list)
        kps_array = []
        person_key_points_dict = self.create_person_keypoints_dict()
        values_dict = {
            "x": "",
            "y": ""
        }

        for n in range(len(person_wise_key_points)):
            for i in range(self.KEY_POINTS_NUMBER - 1):
                pair = self.POSE_PAIRS[i]
                index = person_wise_key_points[n][np.array(pair)]

                if -1 in index:
                    continue

                # b_key_point contains the x coordinates
                # a_key_point contains the y coordinates
                b_key_point = np.int32(key_points_list[index.astype(int), 0])
                a_key_point = np.int32(key_points_list[index.astype(int), 1])

                # Point[0] coordinates = (x = b_key_point[0] , y = a_key_point[0] )
                # Point[1] coordinates = (x = b_key_point[1] , y = a_key_point[1])
                b_key_point_id = str(pair[0])
                values_dict["x"] = int(b_key_point[0])
                values_dict["y"] = int(a_key_point[0])
                person_key_points_dict[b_key_point_id] = values_dict.copy()

                a_key_point_id = str(pair[1])
                values_dict["x"] = int(b_key_point[1])
                values_dict["y"] = int(a_key_point[1])
                person_key_points_dict[a_key_point_id] = values_dict.copy()

                cv2.line(frame_clone, (b_key_point[0], a_key_point[0]),
                         (b_key_point[1], a_key_point[1]), self.KEY_POINT_COLORS[i],
                         3, cv2.LINE_AA)

            kps_array.append(person_key_points_dict.copy())
            person_key_points_dict = self.create_person_keypoints_dict()

        return frame_clone, kps_array

    def draw_key_points(self, frame_clone, detected_key_points):
        # Draw the detected key points in the frame
        for i in range(self.KEY_POINTS_NUMBER):
            for j in range(len(detected_key_points[i])):
                cv2.circle(frame_clone, detected_key_points[i][j][0:2],
                           5, self.KEY_POINT_COLORS[i], -1, cv2.LINE_AA)

    # Find valid connections between the different joints of a all persons present
    def get_valid_pairs(self, output, frame_width, frame_height, detected_key_points):
        valid_pairs = []
        invalid_pairs = []
        n_interp_samples = 15
        paf_score_th = 0.1
        conf_th = 0.7

        # loop for every POSE_PAIR
        for k in range(len(self.MAP_IDX)):
            # A->B constitute a limb
            # paf = Part Affinities: the degree of association between parts
            paf_a = output[0, self.MAP_IDX[k][0], :, :]
            paf_b = output[0, self.MAP_IDX[k][1], :, :]
            paf_a = cv2.resize(paf_a, (frame_width, frame_height))
            paf_b = cv2.resize(paf_b, (frame_width, frame_height))

            # Find the keypoints for the first and second limb
            cand_a = detected_key_points[self.POSE_PAIRS[k][0]]
            cand_b = detected_key_points[self.POSE_PAIRS[k][1]]
            n_a = len(cand_a)
            n_b = len(cand_b)

            # If keypoints for the joint-pair is detected
            # check every joint in candA with every joint in candB
            # Calculate the distance vector between the two joints
            # Find the PAF values at a set of interpolated points between the joints
            # Use the above formula to compute a score to mark the connection valid
            if n_a != 0 and n_b != 0:
                valid_pair = np.zeros((0, 3))
                for i in range(n_a):
                    max_j = -1
                    max_score = -1
                    found = 0
                    for j in range(n_b):
                        # Find d_ij
                        d_ij = np.subtract(cand_b[j][:2], cand_a[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        # Find p(u)
                        interp_coord = list(zip(np.linspace(cand_a[i][0], cand_b[j][0], num=n_interp_samples),
                                                np.linspace(cand_a[i][1], cand_b[j][1], num=n_interp_samples)))
                        # Find L(p(u))
                        paf_interp = []
                        for k in range(len(interp_coord)):
                            paf_interp.append([paf_a[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                               paf_b[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
                        # Find E
                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores) / len(paf_scores)

                        # Check if the connection is valid If the fraction of interpolated vectors aligned with PAF
                        # is higher, then threshold -> Valid Pair
                        if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                            if avg_paf_score > max_score:
                                max_j = j
                                max_score = avg_paf_score
                                found = 1
                    # Append the connection to the list
                    if found:
                        valid_pair = np.append(valid_pair, [[cand_a[i][3], cand_b[max_j][3], max_score]], axis=0)

                # Append the detected connections to the global list
                valid_pairs.append(valid_pair)
            else:  # If no key_points are detected
                # print("No Connection : k = {}".format(k))
                invalid_pairs.append(k)
                valid_pairs.append([])

        return valid_pairs, invalid_pairs

    # This function creates a list of key_points belonging to each person
    # For each detected valid pair, it assigns the joint(s) to a person
    def get_person_wise_key_points(self, valid_pairs, invalid_pairs, key_points_list):
        # the last number in each row is the overall score
        person_wise_key_points = -1 * np.ones((0, 26))

        for k in range(len(self.MAP_IDX)):
            if k not in invalid_pairs:
                part_as = valid_pairs[k][:, 0]
                part_bs = valid_pairs[k][:, 1]
                index_a, index_b = np.array(self.POSE_PAIRS[k])

                for i in range(len(valid_pairs[k])):
                    found = 0
                    person_idx = -1
                    for j in range(len(person_wise_key_points)):
                        if person_wise_key_points[j][index_a] == part_as[i]:
                            person_idx = j
                            found = 1
                            break

                    if found:
                        person_wise_key_points[person_idx][index_b] = part_bs[i]
                        person_wise_key_points[person_idx][-1] += key_points_list[part_bs[i].astype(int), 2] + \
                                                                  valid_pairs[k][i][
                                                                      2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 24:
                        row = -1 * np.ones(26)
                        row[index_a] = part_as[i]
                        row[index_b] = part_bs[i]
                        # add the keypoint_scores for the two key_points and the paf_score
                        row[-1] = sum(key_points_list[valid_pairs[k][i, :2].astype(int), 2]) + valid_pairs[k][i][2]
                        person_wise_key_points = np.vstack([person_wise_key_points, row])

        return person_wise_key_points

    def create_person_keypoints_dict(self):
        base_key_points_dict = {
            "0": "",
            "1": "",
            "2": "",
            "3": "",
            "4": "",
            "5": "",
            "6": "",
            "7": "",
            "8": "",
            "9": "",
            "10": "",
            "11": "",
            "12": "",
            "13": "",
            "14": "",
            "15": "",
            "16": "",
            "17": "",
            "18": "",
            "19": "",
            "20": "",
            "21": "",
            "22": "",
            "23": "",
            "24": ""
        }

        return base_key_points_dict.copy()
