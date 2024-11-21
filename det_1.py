import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2


model = YOLO('yolov10x.pt')
tracker = DeepSort(
    max_age=100,
    n_init=1,
    nn_budget=70
)

crowd_data = []

video_in = '/home/magellan/envs/dynamic_crowd_detection/project_files/vids/crowd_demo.mp4'
video_out = '/home/magellan/envs/dynamic_crowd_detection/project_files/vids/crowd_demo_op.mp4'

cap = cv2.VideoCapture(video_in)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
crowd_clusters = {}

while True:
    num_clusters =0
    response, frame = cap.read()
    if not response:
        break
    frame_count+=1
    results = model.predict(frame)
    cords, cl, cf = [],[],[]
    for det in results[0].boxes.data.tolist():
        if len(det)==6 and (int(det[4])==0 or int(det[4])==1) and int(det[5])==0:
            x1,y1,x2,y2,conf,classs = det
            cords.append([x1,y1,x2,y2])
            cl.append(int(classs))
            cf.append(conf)
    detections = []
    for i in range(len(cords)):
        detections.append([cords[i], cl[i], cf[i]])

    # List[ List[float], List[int or str], List[float] ] List of Polygons, Classes, Confidences. All 3 sublists of the same length. A polygon defined as a ndarray-like [x1,y1,x2,y2,...].
    # detections = [
    # [[x1, y1, x2, y2], class_id, confidence],
    # [[x1, y1, x2, y2], class_id, confidence],
    # ...
    # ]
    tracked_objects = tracker.update_tracks(detections, frame = frame)

    centers = []
    for obj in tracked_objects:
        if not obj.is_confirmed():
            continue
        cen = [(x1+x2)/2 , (y1+y2)/2]
        centers.append(cen)
    
    if len(centers)>=3:
        cluster = DBSCAN(
            eps = 100,
            min_samples=3,   
        ).fit(centers)
        labels = cluster.labels_

        unique_labels = set(labels)
        num_clusters = len(unique_labels-{-1})
        for label in unique_labels:
            if label == -1:  
                continue

            cluster_points  = [centers[i] for i in range(len(labels)) if labels[i] == label]
            if label not in crowd_clusters:
                crowd_clusters[label] = {'frame': frame_count, 'count':len(cluster_points)}
            else:
                crowd_clusters[label]['count'] = len(cluster_points)

            if frame_count - crowd_clusters[label]["frame"] >= 10 and crowd_clusters[label]["count"] >= 3:
                crowd_data.append({"Frame": frame_count, "Crowd Count": crowd_clusters[label]["count"]})
                crowd_clusters[label]["logged"] = True

    for label in list(crowd_clusters.keys()):
        if frame_count - crowd_clusters[label]['frame']>10 and crowd_clusters[label]['logged']==False:
            del crowd_clusters[label]
    
    for obj in tracked_objects:
        if not obj.is_confirmed():
            continue
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    print("FRAME NUMBER = ", frame_count)
    print("PEOPLE DETECTED = ", len(detections))
    print('VALID CLUSTERS = ', num_clusters)
    if len(crowd_data)>=1:
        print("CROWD INFO = ",crowd_data[-1])

    out.write(frame)

df = pd.DataFrame(crowd_data)
df.to_csv('crowd_data', index=False)

cap.release()
out.release()

print('DONE')