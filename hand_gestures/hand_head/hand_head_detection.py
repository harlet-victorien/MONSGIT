import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hands and face detection
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize the solutions
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip horizontally for a selfie-view
    image = cv2.flip(image, 1)
    
    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process for hands and face
    hand_results = hands.process(rgb_image)
    face_results = face_detection.process(rgb_image)
    
    # Variables to store important points
    face_center = None
    head_size = 0
    hand_centers = []
    hand_types = []
    
    # Handle face detection
    if face_results.detections:
        for detection in face_results.detections:
            # Draw face detection box
            mp_drawing.draw_detection(image, detection)
            
            # Get bounding box
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            
            # Calculate face center and size
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            
            face_center = (int(x + w/2), int(y + h/2))
            head_size = max(w, h)  # Use the larger dimension as head size reference
            
            # Draw face center
            cv2.circle(image, face_center, 5, (255, 0, 0), -1)
    
    # Handle hand landmarks
    if hand_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            # Determine if it's left or right hand
            handedness = hand_results.multi_handedness[idx].classification[0].label
            hand_type = "Left" if handedness == "Right" else "Right"  # Flipped because image is mirrored
            hand_types.append(hand_type)
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Use wrist as hand center
            wrist = hand_landmarks.landmark[0]
            ih, iw, _ = image.shape
            wrist_pos = (int(wrist.x * iw), int(wrist.y * ih))
            hand_centers.append(wrist_pos)
            cv2.circle(image, wrist_pos, 5, (0, 255, 0), -1)
    
    # Calculate and display distances if both face and hands are detected
    if face_center and hand_centers and head_size > 0:
        for idx, hand_center in enumerate(hand_centers):
            # Calculate distance
            distance = calculate_distance(face_center, hand_center)
            
            # Normalize by head size
            normalized_distance = distance / head_size
            
            # Draw line between hand and face
            cv2.line(image, face_center, hand_center, (0, 0, 255), 2)
            
            # Display the normalized distance
            text_position = (hand_center[0], hand_center[1] - 10)
            cv2.putText(image, f"{hand_types[idx]}: {normalized_distance:.2f}", 
                      text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Display
    cv2.imshow('Hand and Face Detection with Distance', image)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()