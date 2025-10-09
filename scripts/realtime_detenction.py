import cv2
import numpy as np
import joblib
from ultralytics import YOLO
from pydantic import BaseModel
import os

class GetKeypoint(BaseModel):
    NOSE:           int = 0; LEFT_EYE:       int = 1; RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3; RIGHT_EAR:      int = 4; LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6; LEFT_ELBOW:     int = 7; RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9; RIGHT_WRIST:    int = 10; LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12; LEFT_KNEE:      int = 13; RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15; RIGHT_ANKLE:    int = 16

class DetectKeypoint:
    def __init__(self, yolov8_model='yolov8m-pose.pt'):
        self.get_keypoint = GetKeypoint()
        self.model = YOLO(yolov8_model)

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
        min_distance, central_keypoint = float('inf'), None
        for i, box in enumerate(results.boxes.xyxy):
            box_center_x, box_center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            distance = np.sqrt((box_center_x - img_center_x)**2 + (box_center_y - img_center_y)**2)
            if distance < min_distance:
                min_distance = distance
                central_keypoint = results.keypoints.xyn.cpu().numpy()[i]
        return self.extract_keypoint(central_keypoint) if central_keypoint is not None else {}

    def __call__(self, image: np.ndarray):
        return self.model(image, verbose=False)[0]

def flatten_keypoints(keypoints: dict) -> list:
    return [coord for key in keypoints for coord in keypoints[key]]


print("Carregando modelos...")
model_dir = '../models/'
pose_classifier = joblib.load(os.path.join(model_dir, 'pose_classifier_model.joblib'))
label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
keypoint_detector = DetectKeypoint('yolov8m-pose.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível abrir a webcam.")
    exit()

print("Iniciando detecção em tempo real... Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = keypoint_detector(frame)
    keypoints = keypoint_detector.get_central_keypoint(results, frame.shape)
    
    pose_name = "N/A"
    
    if keypoints:
        # Achata os keypoints para o formato de entrada do modelo
        input_data = np.array(flatten_keypoints(keypoints)).reshape(1, -1)
        
        # Faz a predição
        prediction_encoded = pose_classifier.predict(input_data)
        
        # Decodifica para o nome da pose
        pose_name = label_encoder.inverse_transform(prediction_encoded)[0]

    cv2.putText(frame, f"POSE: {pose_name.upper()}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    if keypoints:
        for key, (x,y) in keypoints.items():
             if x is not None and y is not None:
                x_int, y_int = int(x * frame.shape[1]), int(y * frame.shape[0])
                cv2.circle(frame, (x_int, y_int), 5, (0, 0, 255), -1)

    cv2.imshow('Reconhecimento de Pose', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Detecção finalizada.")

