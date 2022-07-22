import torch
import cv2
import numpy as np
import pandas as pd


COLORS = np.random.uniform(0, 255, size=(80, 3))
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, classes=80)


def draw_boxes(results, image):

    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, row in results.iterrows():
        color = COLORS[i]
        xmin, xmax, ymin, ymax = row[['xmin', 'xmax', 'ymin', 'ymax']]
        cv2.rectangle(
            image,
            (int(xmin), int(ymin)),
            (int(xmax), int(ymax)),
            color, 2
        )
        cv2.putText(image, row["name"], (int(xmin), int(ymin-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Cannot connect camera')

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv2.putText(frame, f"Test", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        with torch.no_grad():
            out = yolo_model(frame)
        yolo_results = out.pandas().xyxy[0]

        frame = draw_boxes(yolo_results, frame)

        cv2.imshow('image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break
