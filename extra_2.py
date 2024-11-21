import cv2
from ultralytics import YOLO

model = YOLO('yolov10x.pt')

video_in = '/home/magellan/envs/dynamic_crowd_detection/project_files/vids/crowd_demo.mp4'
video_out = '/home/magellan/envs/dynamic_crowd_detection/project_files/vids/crowd_demo_model_test_x.mp4'

cap = cv2.VideoCapture(video_in)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while True:
    response, frame = cap.read()
    if not response:
        break
    results = model.predict(frame)

    for det in results[0].boxes.data.tolist():
        x1,y1,x2,y2,conf,classs = det
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    out.write(frame)

cap.release()
out.release()