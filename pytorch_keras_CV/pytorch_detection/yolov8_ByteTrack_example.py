import cv2
from ultralytics import YOLO


# train custom detector with: model.train(data=f"detect_dataset.yaml", epochs=3) and .yaml config
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
# yolov8 to detect learned objects, then apply ByteTrack to track these objects.
# Sequential similarity scores for high confident detections and low confident detections
# Kalman filter to track and cost-based ReID to update existing/new/disappeared objects

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, tracker="bytetrack.yaml", persist=True)  # or 'botsort.yaml'
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    for box, id in zip(boxes, ids):
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Id {id}",
            (box[0], box[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
