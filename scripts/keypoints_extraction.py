import os
import csv
import sys
import cv2
import numpy as np
from pydantic import BaseModel
from ultralytics import YOLO

class GetKeypoint(BaseModel):
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16

class DetectKeypoint:
    def __init__(self, yolov8_model='yolov8m-pose.pt'):
        self.yolov8_model = yolov8_model
        self.get_keypoint = GetKeypoint()
        self.model = YOLO(self.yolov8_model)

    def extract_keypoint(self, keypoint: np.ndarray) -> dict:
        return {
            'nose': keypoint[self.get_keypoint.NOSE], 'left_eye': keypoint[self.get_keypoint.LEFT_EYE],
            'right_eye': keypoint[self.get_keypoint.RIGHT_EYE], 'left_ear': keypoint[self.get_keypoint.LEFT_EAR],
            'right_ear': keypoint[self.get_keypoint.RIGHT_EAR], 'left_shoulder': keypoint[self.get_keypoint.LEFT_SHOULDER],
            'right_shoulder': keypoint[self.get_keypoint.RIGHT_SHOULDER], 'left_elbow': keypoint[self.get_keypoint.LEFT_ELBOW],
            'right_elbow': keypoint[self.get_keypoint.RIGHT_ELBOW], 'left_wrist': keypoint[self.get_keypoint.LEFT_WRIST],
            'right_wrist': keypoint[self.get_keypoint.RIGHT_WRIST], 'left_hip': keypoint[self.get_keypoint.LEFT_HIP],
            'right_hip': keypoint[self.get_keypoint.RIGHT_HIP], 'left_knee': keypoint[self.get_keypoint.LEFT_KNEE],
            'right_knee': keypoint[self.get_keypoint.RIGHT_KNEE], 'left_ankle': keypoint[self.get_keypoint.LEFT_ANKLE],
            'right_ankle': keypoint[self.get_keypoint.RIGHT_ANKLE],
        }
    
    def get_central_keypoint(self, results, img_shape):
        if results.keypoints is None or results.keypoints.xyn is None or results.keypoints.xyn.shape[0] == 0:
            return {}
        
        img_center_x, img_center_y = img_shape[1] / 2, img_shape[0] / 2
        min_distance = float('inf')
        central_keypoint = None

        for i, box in enumerate(results.boxes.xyxy):
            box_center_x = (box[0] + box[2]) / 2
            box_center_y = (box[1] + box[3]) / 2
            distance = np.sqrt((box_center_x - img_center_x)**2 + (box_center_y - img_center_y)**2)
            if distance < min_distance:
                min_distance = distance
                central_keypoint = results.keypoints.xyn.cpu().numpy()[i]
        
        return self.extract_keypoint(central_keypoint) if central_keypoint is not None else {}

    def __call__(self, image: np.ndarray):
        return self.model(image)[0]
    
    def flatten_keypoints(keypoints: dict) -> list:
        return [coord for key in keypoints for coord in keypoints[key]]

    def has_valid_keypoints(keypoints: dict) -> bool:
        required_indices = list(range(13)) # 0 a 12 (até os quadris)
        keypoint_names = list(keypoints.keys())
        for i in required_indices:
            key = keypoint_names[i]
            x, y = keypoints[key]
            if x == 0 or y == 0 or np.isnan(x) or np.isnan(y):
                return False
        return True
    
    def process_directories(directories, output_csv):
        detector = DetectKeypoint('yolov8m-pose.pt')
        header = ['filename', 'pose_category'] + [f'{name}_{axis}' for name in GetKeypoint.__annotations__ for axis in ['x', 'y']]
        
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            
            for directory, pose_category in directories:
                print(f"Processando pasta: {directory}")
                for filename in os.listdir(directory):
                    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(directory, filename)
                        img = cv2.imread(img_path)
                        if img is None: continue
                        
                        results = detector(img)
                        keypoints = detector.get_central_keypoint(results, img.shape)
                        
                        if keypoints and has_valid_keypoints(keypoints):
                            row = [filename, pose_category] + flatten_keypoints(keypoints)
                            writer.writerow(row)
                print(f"Pasta {directory} finalizada.")

if __name__ == "__main__":
    base_dataset_path = '../dataset/'
    output_csv_path = '../models/keypoints_dataset.csv'
    
    directories = [
        (os.path.join(base_dataset_path, 'stop'), 'stop'),
        (os.path.join(base_dataset_path, 'nao_acao'), 'nao_acao')
    ]
    
    process_directories(directories, output_csv_path)
    print(f"\nExtração concluída! Dataset salvo em: {output_csv_path}")

