import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hands and face mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize the solutions
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                max_num_faces=1,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to check if a point is inside a rectangle
def is_point_in_rect(point, rect_x, rect_y, rect_w, rect_h):
    x, y = point
    return rect_x <= x <= rect_x + rect_w and rect_y <= y <= rect_y + rect_h

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
    face_results = face_mesh.process(rgb_image)
    
    # Variables to store important points
    face_center = None
    head_size = 0
    hand_centers = []
    hand_types = []
    
    # Process hand landmarks FIRST to ensure they're available for square detection
    if hand_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            # Determine if it's left or right hand
            handedness = hand_results.multi_handedness[idx].classification[0].label
            hand_type = "Left" if handedness == "Right" else "Right"  # Flipped because image is mirrored
            hand_types.append(hand_type)
            
            # Use wrist as hand center
            ih, iw, _ = image.shape
            wrist = hand_landmarks.landmark[0]
            wrist_pos = (int(wrist.x * iw), int(wrist.y * ih))
            hand_centers.append(wrist_pos)
            
            # Store index finger tip position as well for additional detection point
            index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip
            index_pos = (int(index_finger_tip.x * iw), int(index_finger_tip.y * ih))
            hand_centers.append(index_pos)
            hand_types.append(f"{hand_type}_tip")
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Mark wrist center
            cv2.circle(image, wrist_pos, 5, (0, 255, 0), -1)
            cv2.circle(image, index_pos, 5, (255, 255, 0), -1)  # Yellow for finger tip
    
    # Handle face mesh landmarks
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # No need to draw the full mesh
            
            # Get important face landmarks
            ih, iw, _ = image.shape
            
            # Use nose tip as face center (landmark 4)
            nose_tip = face_landmarks.landmark[4]
            face_center = (int(nose_tip.x * iw), int(nose_tip.y * ih))
            
            # Draw nose point with larger circle for better visibility
            cv2.circle(image, face_center, 8, (0, 0, 255), -1)
            
            # Calculate head size (using distance between eyes)
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            left_eye_pos = (int(left_eye.x * iw), int(left_eye.y * ih))
            right_eye_pos = (int(right_eye.x * iw), int(right_eye.y * ih))
            head_size = calculate_distance(left_eye_pos, right_eye_pos)
            
            # Optionally, also mark eye positions but smaller than nose
            # cv2.circle(image, left_eye_pos, 3, (0, 255, 255), -1)
            # cv2.circle(image, right_eye_pos, 3, (0, 255, 255), -1)
            
            # Define squares at normalized positions around the head
            square_size = int(head_size * 4)  # Size of squares relative to head size
            distance_factor = 1.5  # Distance from head center as a factor of head_size
            
            squares = [
                # Left square
                {
                    'x': int(face_center[0] - distance_factor * head_size - square_size),
                    'y': int(face_center[1] - square_size/2),
                    'name': 'Left Zone'
                },
                # Right square
                {
                    'x': int(face_center[0] + distance_factor * head_size),
                    'y': int(face_center[1] - square_size/2),
                    'name': 'Right Zone'
                }
            ]
            
            # Draw the squares and check for hands inside them
            for square in squares:
                x, y = square['x'], square['y']
                name = square['name']
                
                # Initialize with inactive color (red)
                color = (0, 0, 255)  # Red by default
                hand_in_square = False
                which_hand = ""
                
                # Make squares more visible with semi-transparent fill
                overlay = image.copy()
                cv2.rectangle(overlay, (x, y), (x + square_size, y + square_size), (0, 0, 255), -1)
                alpha = 0.2  # Transparency factor
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                
                # Check if any hand is inside this square
                for idx, hand_center in enumerate(hand_centers):
                    if is_point_in_rect(hand_center, x, y, square_size, square_size):
                        color = (0, 255, 0)  # Green if hand is inside
                        hand_in_square = True
                        which_hand = hand_types[idx]
                        
                        # Make square green with semi-transparent fill when hand is inside
                        overlay = image.copy()
                        cv2.rectangle(overlay, (x, y), (x + square_size, y + square_size), (0, 255, 0), -1)
                        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                        break
                
                # Draw the square outline
                cv2.rectangle(image, (x, y), (x + square_size, y + square_size), color, 2)
                
                # Display zone name and hand detection status
                status_text = f"{name}: {which_hand}" if hand_in_square else f"{name}"
                cv2.putText(image, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, color, 2)
    
    # Calculate and display distances if both face and hands are detected
    if face_center and len(hand_centers) > 0 and head_size > 0:
        for idx, hand_center in enumerate(hand_centers):
            if "tip" not in hand_types[idx]:  # Only draw lines for wrists, not fingertips
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
    cv2.imshow('Hand and Nose Tracking with Interaction Zones', image)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()