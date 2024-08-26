import torch
import torchvision
import torchvision.transforms as T
from threading import Thread
import cv2

def StartModelAndCap():
        # Load YOLOv5 model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='G:\\MazeGameIRLSheepRL- LowResTest no Threading\yolov5Small\\runs\\train\\exp\\weights\\best.pt')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device).eval()
        return model, device

def geoCoordinates(frame, model, device, transform):
    # Perform inference
    results = model(frame)
    detections = results.pandas().xyxy[0]

    # Initialize variables to store the highest confidence detections
    highest_conf_ball = None
    highest_conf_qr = None
    highest_conf_ball_score = 0
    highest_conf_qr_score = 0

    # Iterate through the detections
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        class_id = int(row['class'])

        if class_id == 0:  # Assuming class 0 is for the ball
            if confidence > highest_conf_ball_score:
                highest_conf_ball = [x1, y1, x2, y2]
                highest_conf_ball_score = confidence
        elif class_id == 1:  # Assuming class 1 is for QR code
            if confidence > highest_conf_qr_score:
                highest_conf_qr = [x1, y1, x2, y2]
                highest_conf_qr_score = confidence
    

    return highest_conf_qr, highest_conf_ball


