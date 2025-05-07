import cv2
from ultralytics import YOLO

# Load YOLOv8 model (use 'yolov8n.pt' for a fast, lightweight version)
model = YOLO("yolov8n.pt")  # You can also use 'yolov8s.pt', 'yolov8m.pt', etc.

# Open webcam (0 is default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Run object detection
    results = model(frame, stream=True)

    # Draw results
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLOv8 Webcam Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release and destroy
cap.release()
cv2.destroyAllWindows()
