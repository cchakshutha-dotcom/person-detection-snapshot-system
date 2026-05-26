from ultralytics import YOLO
import cv2, os
import requests
from datetime import datetime

# Load model (auto download if not present)
model = YOLO('yolov8n')

# Open webcam
cap = cv2.VideoCapture(0)

# Create folder
os.makedirs('snapshots', exist_ok=True)

last_save = 0

# Get location once (faster)
def get_location_by_ip():
    try:
        response = requests.get('https://ipapi.co/json')
        if response.status_code == 200:
            data = response.json()
            return {
                'city': data.get('city', 'unknown'),
                'country': data.get('country_name', 'unknown')
            }
    except:
        pass
    return {'city': 'unknown', 'country': 'unknown'}

location = get_location_by_ip()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect ALL objects
    results = model(frame)[0]

    # Draw detections
    frame = results.plot()

    # Count objects
    counts = {}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        counts[label] = counts.get(label, 0) + 1

    # Convert counts to text
    y = 30
    for label, count in counts.items():
        text = f"{label}: {count}"
        cv2.putText(frame, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        y += 30

    # Save snapshot every 5 sec if objects detected
    if len(counts) > 0 and (datetime.now().timestamp() - last_save) >= 5:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        cv2.putText(frame,
                    f"{location['city']}, {location['country']}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        cv2.imwrite(f"snapshots/{timestamp}.jpg", frame)
        print("Snapshot saved")

        last_save = datetime.now().timestamp()

    # Show window
    cv2.imshow("YOLOv8 All Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()