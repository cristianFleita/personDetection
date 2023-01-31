class KeyPointConfig:
    
    def __init__(self, model_id):
        if (model_id == "body_25"):
            self.initialize_body_25_model()
        elif (model_id == "coco_18"):
            self.initialize_coco_18_model()

    def initialize_body_25_model(self):
        self.PROTO_FILE = "pose/body_25/pose_deploy.prototxt"
        self.WEIGHTS_FILE = "pose/body_25/pose_iter_584000.caffemodel"
        self.N_POINTS = 25

        self.KEYPOINTS_MAPPING = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye","REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]

        self.POSE_PAIRS = [[1,8], [1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
            [8,9], [9,10], [10,11], [8,12], [12,13], [13,14],
            [1,0], [0,15], [15,17], [0,16], [16,18], [2,17], [5,18],
            [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]]

        self.MAP_INDEXES = [[26, 27], [40, 41], [48, 49], [42, 43], [44, 45], [50, 51],
            [52, 53], [32, 33], [28, 29], [30, 31], [34, 35], [36, 37],
            [38, 39], [56, 57], [58, 59], [62, 63], [60, 61], [64, 65],
            [46, 47], [54, 55], [66, 67], [68, 69], [70, 71], [72, 73],
            [74, 75], [76, 77]]

        self.COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
            [255, 255, 0], [170, 255, 0], [85, 255, 0],
            [0, 255, 0], [0, 255, 85], [0, 255, 170],
            [0, 255, 255], [0, 170, 255], [0, 85, 255],
            [0, 0, 255], [85, 0, 255], [170, 0, 255],
            [255, 0, 255], [255, 0, 170], [255, 0, 85],
            [255, 170, 85], [255, 170, 170], [255, 170, 255],
            [255, 85, 85], [255, 85, 170], [255, 85, 255],
            [170, 170, 170]]

    def initialize_coco_18_model(self):
        self.PROTO_FILE = "pose/coco/pose_deploy_linevec.prototxt"
        self.WEIGHTS_FILE = "pose/coco/pose_iter_440000.caffemodel"

    def get_proto_file(self):
        return self.PROTO_FILE

    def get_weights_file(self):
        return self.WEIGHTS_FILE
    
    def get_keypoints_mapping(self):
        return self.KEYPOINTS_MAPPING

    def get_pose_pairs(self):
        return self.POSE_PAIRS
    
    def get_map_indexes(self):
        return self.MAP_INDEXES
    
    def get_colors(self):
        return self.COLORS

    def get_n_points(self):
        return self.N_POINTS