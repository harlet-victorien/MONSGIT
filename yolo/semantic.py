import cv2
from ultralytics import YOLO

# Load YOLOv8 segmentation model (n = nano, s = small, etc.)
model = YOLO("yolov8n-seg.pt")  # You can use 'yolov8s-seg.pt' for better quality

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Run segmentation inference
    results = model(frame, stream=True)

    # Draw the masks and results on the frame
    for result in results:
        frame = result.plot()  # Includes masks, boxes, and labels

    # Display the result
    cv2.imshow("YOLOv8 Instance Segmentation", frame)

    # Exit when 'esc' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
