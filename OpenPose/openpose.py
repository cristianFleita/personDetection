import cv2
import json
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on")

args = parser.parse_args()

protoFile = "pose/body_25/pose_deploy.prototxt"
weightsFile = "pose/body_25/pose_iter_584000.caffemodel"

nPoints = 25

keypointsMapping = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye","REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]

POSE_PAIRS = [[1,8], [1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
            [8,9], [9,10], [10,11], [8,12], [12,13], [13,14],
            [1,0], [0,15], [15,17], [0,16], [16,18], [2,17], [5,18],
            [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]]

mapIdx =[[26, 27], [40, 41], [48, 49], [42, 43], [44, 45], [50, 51],
            [52, 53], [32, 33], [28, 29], [30, 31], [34, 35], [36, 37],
            [38, 39], [56, 57], [58, 59], [62, 63], [60, 61], [64, 65],
            [46, 47], [54, 55], [66, 67], [68, 69], [70, 71], [72, 73],
            [74, 75], [76, 77]]


colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
            [255, 255, 0], [170, 255, 0], [85, 255, 0],
            [0, 255, 0], [0, 255, 85], [0, 255, 170],
            [0, 255, 255], [0, 170, 255], [0, 85, 255],
            [0, 0, 255], [85, 0, 255], [170, 0, 255],
            [255, 0, 255], [255, 0, 170], [255, 0, 85],
            [255, 170, 85], [255, 170, 170], [255, 170, 255],
            [255, 85, 85], [255, 85, 170], [255, 85, 255],
            [170, 170, 170]]

def getKeypoints(probMap, threshold=0.1):
    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
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

# Find valid connections between the different joints of a all persons present
def getValidPairs(output, frameWidth, frameHeight, detected_keypoints):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 15
    paf_score_th = 0.1
    conf_th = 0.7

    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid
        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
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
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                        pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])

    return valid_pairs, invalid_pairs

# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 26))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

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
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 24:
                    row = -1 * np.ones(26)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    
    return personwiseKeypoints

def process_keypoints(image1, filename):
    frameWidth = image1.shape[1]
    frameHeight = image1.shape[0]

    t = time.time()
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    if args.device == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif args.device == "gpu":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")

    # Fix the input Height and get the width according to the Aspect Ratio
    inHeight = 368
    inWidth = int((inHeight/frameHeight)*frameWidth)

    inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                            (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()
    print("Time Taken in forward pass = {}".format(time.time() - t))

    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    threshold = 0.1

    for part in range(nPoints):
        probMap = output[0,part,:,:]
        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
        # print("ProbMap = {}".format(probMap))

        keypoints = getKeypoints(probMap, threshold)

        # name_part = keypointsMapping[part]
        # print("{} : {}".format(keypointsMapping[part], keypoints))
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)

    (frameClone , kps_array) = draw_keypoints(image1, frameWidth, frameHeight, output, detected_keypoints, keypoints_list)
    #Create the output file name by removing the '.jpg' part
    save_result(frameClone, filename)

    return kps_array

def save_result(frameClone, filename):
    size = len(filename)
    new_filename = filename[:size - 4]
    new_filename = new_filename + '_detect.jpg'
    cv2.imwrite( new_filename, frameClone)

def draw_keypoints(image1, frameWidth, frameHeight, output, detected_keypoints, keypoints_list):
    frameClone = image1.copy()

    #Draw the keypoints
    for i in range(nPoints):
        for j in range(len(detected_keypoints[i])):
            cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
    # cv2.imshow("Keypoints",frameClone)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    valid_pairs, invalid_pairs = getValidPairs(output, frameWidth, frameHeight, detected_keypoints)
    personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)
    kps_array = []
    person_keypoints_dict = create_person_keypoints_dict()
    values_dict = {
                    "x":"",
                    "y":""
                    }

    for n in range(len(personwiseKeypoints)):
        for i in range(nPoints-1):
            # print("KP:", POSE_PAIRS[i])
            pair = POSE_PAIRS[i]
            index = personwiseKeypoints[n][np.array(pair)]

            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            
            B_key_point_id = str(pair[0])
            values_dict["x"] = int (B[0])
            values_dict["y"] = int (A [0])

            person_keypoints_dict[B_key_point_id] = values_dict.copy()

            A_key_point_id = str(pair[1])
            values_dict["x"] = int (B[1])
            values_dict["y"] = int (A [1])

            person_keypoints_dict[A_key_point_id] = values_dict.copy()

            # print("A= y coord:",A)
            # print("B:= x cooord",B)

            # pair[0] = [ (x = B[0] , y = A [0] ) ]
            # pair[1] = [ (x = B[1] , y = A [1] ) ]
            
            # print(pair[0])
            # print(pair[1])

            cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
        
        kps_array.append (person_keypoints_dict.copy())

    # cv2.imshow("Detected Pose" , frameClone)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    return frameClone , kps_array

def create_person_keypoints_dict():
    dict = {
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

    return dict.copy()