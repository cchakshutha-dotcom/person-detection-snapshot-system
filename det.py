from ultralytics import YOLO
import cv2
import torch

model=YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

cv2.namedWindow('YOLO object detection', cv2.WINDOW_NORMAL)

# Set window to fullscreen
cv2.setWindowProperty(
    'YOLO object detection',
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)



while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO object detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()