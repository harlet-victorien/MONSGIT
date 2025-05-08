import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import keyboard

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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get screen dimensions for mouse control
screen_width, screen_height = pyautogui.size()

# Variables to track click states and control enable/disable
clicked_left = False
clicked_middle = False
clicked_right = False
clicked_scroll = False
control_enabled = True

def toggle_control(event):
    global control_enabled
    control_enabled = not control_enabled
    print(f"Control {'enabled' if control_enabled else 'disabled'}")

# Register the key press event to toggle control
keyboard.on_press_key('n', toggle_control)

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to check if a point is inside a rectangle
def is_point_in_rect(point, rect_x, rect_y, rect_w, rect_h):
    x, y = point
    return rect_x <= x <= rect_x + rect_w and rect_y <= y <= rect_y + rect_h

# Function to check if two fingers are together, with normalized distance
def are_fingers_together(landmarks, finger1_tip_id, finger2_tip_id):
    tip1 = landmarks.landmark[finger1_tip_id]
    tip2 = landmarks.landmark[finger2_tip_id]
    
    # Calculate Euclidean distance in 2D space (x and y) between the two fingertips
    tips_distance = ((tip1.x - tip2.x) ** 2 + (tip1.y - tip2.y) ** 2) ** 0.5
    
    # Use wrist to index fingertip distance as a normalizing factor
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Calculate hand size reference (wrist to index tip distance)
    hand_size_reference = ((wrist.x - index_tip.x) ** 2 + (wrist.y - index_tip.y) ** 2) ** 0.5
    
    # Normalize the distance by dividing by the hand size reference
    normalized_distance = tips_distance / hand_size_reference if hand_size_reference > 0 else float('inf')
    
    # Define a threshold for normalized "togetherness"
    threshold = 0.2  # Adjust as needed based on testing
    
    return normalized_distance < threshold

# Function to check if finger is down (tip below base)
def is_finger_down(landmarks, finger_tip_id, finger_mcp_id):
    return landmarks.landmark[finger_tip_id].y > landmarks.landmark[finger_mcp_id].y

# Function to check if finger is up (tip above base)
def is_finger_up(landmarks, finger_tip_id, finger_mcp_id):
    return landmarks.landmark[finger_tip_id].y < landmarks.landmark[finger_mcp_id].y

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
    hand_landmarks_dict = {}  # To store hand landmarks for each hand type
    
    # Display control status
    status_text = "Control: ENABLED" if control_enabled else "Control: DISABLED"
    cv2.putText(
        image, 
        status_text, 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )
    
    # Process hand landmarks FIRST to ensure they're available for square detection
    if hand_results.multi_hand_landmarks:
        # Display how many hands are detected
        num_hands = len(hand_results.multi_hand_landmarks)
        cv2.putText(
            image, 
            f"Hands detected: {num_hands}", 
            (10, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            # Determine if it's left or right hand
            handedness = hand_results.multi_handedness[idx].classification[0].label
            hand_type = "Left" if handedness == "Right" else "Right"  # Flipped because image is mirrored
            hand_types.append(hand_type)
            
            # Store landmarks for later use in gesture detection
            hand_landmarks_dict[hand_type] = hand_landmarks
            
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
            
            # Define squares at normalized positions around the head
            square_size = int(head_size * 4)  # Size of squares relative to head size
            distance_factor = 1.5  # Distance from head center as a factor of head_size
            
            squares = [
                # Left square
                {
                    'x': int(face_center[0] - distance_factor * head_size - square_size),
                    'y': int(face_center[1] - square_size/2),
                    'name': 'Right Zone',
                    'hand_detected': None
                },
                # Right square
                {
                    'x': int(face_center[0] + distance_factor * head_size),
                    'y': int(face_center[1] - square_size/2),
                    'name': 'Left Zone',
                    'hand_detected': None
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
                
                # Check for hands where both wrist and index finger are inside the square
                if len(hand_centers) >= 2:  # Make sure we have at least some hand points
                    for i in range(0, len(hand_centers), 2):  # Process pairs of points (wrist and index)
                        if i+1 < len(hand_centers):  # Make sure we have both wrist and index
                            wrist_idx = i
                            index_idx = i+1
                            
                            # Check if both points are in the square
                            wrist_in_square = is_point_in_rect(hand_centers[wrist_idx], x, y, square_size, square_size)
                            index_in_square = is_point_in_rect(hand_centers[index_idx], x, y, square_size, square_size)
                            
                            # Only consider hand in square if both wrist and index finger are inside
                            if wrist_in_square and index_in_square:
                                color = (0, 255, 0)  # Green if hand is inside
                                hand_in_square = True
                                
                                # Get the hand type (remove "_tip" suffix if present)
                                base_hand_type = hand_types[wrist_idx]
                                which_hand = base_hand_type
                                
                                # Store which hand is in this square
                                square['hand_detected'] = base_hand_type
                                
                                # Make square green with semi-transparent fill when hand is inside
                                overlay = image.copy()
                                cv2.rectangle(overlay, (x, y), (x + square_size, y + square_size), (0, 255, 0), -1)
                                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                                break  # Exit loop if hand is found in square
        
                # Draw the square outline
                cv2.rectangle(image, (x, y), (x + square_size, y + square_size), color, 2)
                
                # Display zone name and hand detection status
                status_text = f"{name}: {which_hand}" if hand_in_square else f"{name}"
                cv2.putText(image, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, color, 2)
            
            # Check if both hands are in their respective squares
            left_hand_in_left_square = False
            right_hand_in_right_square = False
            
            for square in squares:
                if square['name'] == 'Left Zone' and square['hand_detected'] == 'Left':
                    left_hand_in_left_square = True
                elif square['name'] == 'Right Zone' and square['hand_detected'] == 'Right':
                    right_hand_in_right_square = True
            
            # Only enable control actions if both hands are in their correct squares
            if control_enabled and left_hand_in_left_square and right_hand_in_right_square:
                # Calculate and move cursor
                if 'Left' in hand_landmarks_dict and 'Right' in hand_landmarks_dict:
                    # Get left hand landmarks for gesture detection
                    left_hand = hand_landmarks_dict['Left']
                    right_hand = hand_landmarks_dict['Right']
                    
                    # Check if index and middle fingers are together on the right hand
                    fingers_together = are_fingers_together(
                        right_hand, 
                        mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                    )
                    
                    # Display status on image
                    cv2.putText(
                        image, 
                        "CONTROL ACTIVE - Both hands in position", 
                        (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 255), 
                        2
                    )
                    
                    if fingers_together:
                        # Find which square contains the left hand
                        left_square = None
                        for sq in squares:
                            if sq['hand_detected'] == 'Left':
                                left_square = sq
                                break
                        
                        if left_square:
                            # Get left hand's position
                            base_middle = left_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                            ih, iw, _ = image.shape
                            hand_pos_x = int(base_middle.x * iw)
                            hand_pos_y = int(base_middle.y * ih)
                            
                            # Get square boundaries
                            sq_x = left_square['x']
                            sq_y = left_square['y']
                            
                            # Calculate position relative to the square (0 to 1)
                            rel_x = max(0, min(1, (hand_pos_x - sq_x) / square_size))
                            rel_y = max(0, min(1, (hand_pos_y - sq_y) / square_size))
                            
                            # Display relative position for debugging
                            cv2.putText(
                                image, 
                                f"Rel Pos: ({rel_x:.2f}, {rel_y:.2f})", 
                                (10, 350), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (255, 255, 255), 
                                1
                            )
                            
                            # Map to screen coordinates
                            cursor_x = int(rel_x * screen_width)
                            cursor_y = int(rel_y * screen_height)
                            
                            # Move the mouse cursor
                            pyautogui.moveTo(cursor_x, cursor_y)
                            
                            # Indicate that we're moving the mouse
                            cv2.putText(
                                image, 
                                f"Moving mouse: {cursor_x}, {cursor_y}", 
                                (10, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (0, 0, 255), 
                                2
                            )
                        else:
                            cv2.putText(
                                image, 
                                "Left hand not in a square", 
                                (10, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (0, 0, 255), 
                                2
                            )
                        
                        # Rest of the code (click functions, etc.) remains the same
                        # Check for middle finger down gesture (left click)
                        middle_finger_down = is_finger_down(
                            left_hand,
                            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                            mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                        )
                        
                        if middle_finger_down:
                            if not clicked_left:
                                pyautogui.click()
                                cv2.putText(
                                    image, 
                                    "Left click", 
                                    (10, 190), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, 
                                    (255, 0, 0), 
                                    2
                                )
                                clicked_left = True
                        else:
                            clicked_left = False
                        
                        # Check for index finger up gesture (middle click hold/release)
                        index_finger_up = is_finger_up(
                            left_hand,
                            mp_hands.HandLandmark.INDEX_FINGER_TIP,
                            mp_hands.HandLandmark.INDEX_FINGER_MCP
                        )
                        
                        if not index_finger_up:  # Index finger is down
                            if not clicked_middle:
                                pyautogui.mouseDown(button='middle')
                                cv2.putText(
                                    image, 
                                    "Middle click hold", 
                                    (10, 230), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, 
                                    (255, 0, 0), 
                                    2
                                )
                                clicked_middle = True
                        else:
                            if clicked_middle:
                                pyautogui.mouseUp(button='middle')
                                clicked_middle = False
                        
                        # Check for ring finger down gesture (right click hold/release)
                        ring_finger_down = is_finger_down(
                            left_hand,
                            mp_hands.HandLandmark.RING_FINGER_TIP,
                            mp_hands.HandLandmark.RING_FINGER_MCP
                        )
                        
                        if ring_finger_down:
                            if not clicked_right:
                                pyautogui.mouseDown(button='right')
                                cv2.putText(
                                    image, 
                                    "Right click hold", 
                                    (10, 270), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, 
                                    (255, 0, 0), 
                                    2
                                )
                                clicked_right = True
                        else:
                            if clicked_right:
                                pyautogui.mouseUp(button='right')
                                clicked_right = False
                        
                        # Compute the distance between thumb and pinky for scrolling
                        thumb_pinky_distance = ((left_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x - 
                                                left_hand.landmark[mp_hands.HandLandmark.PINKY_TIP].x) ** 2 +
                                            (left_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y - 
                                            left_hand.landmark[mp_hands.HandLandmark.PINKY_TIP].y) ** 2) ** 0.5
                        
                        # Use wrist to index fingertip distance as a normalizing factor
                        wrist = left_hand.landmark[mp_hands.HandLandmark.WRIST]
                        index_tip = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        hand_size_reference = ((wrist.x - index_tip.x) ** 2 + (wrist.y - index_tip.y) ** 2) ** 0.5
                        
                        # Normalize the distance
                        normalized_distance = thumb_pinky_distance / hand_size_reference if hand_size_reference > 0 else float('inf')
                        
                        # Define thresholds for scrolling
                        scroll_up_threshold = 0.2  # Adjust based on testing
                        
                        if normalized_distance < scroll_up_threshold:  # Thumb and pinky are close
                            pyautogui.scroll(75)
                            cv2.putText(
                                image, 
                                "Scroll up", 
                                (10, 310), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (255, 0, 0), 
                                2
                            )
                        elif normalized_distance > 0.2 and normalized_distance < 0.5:  # Thumb and pinky are far
                            pyautogui.scroll(-75)
                            cv2.putText(
                                image, 
                                "Scroll down", 
                                (10, 310), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (255, 0, 0), 
                                2
                            )
    
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
    cv2.imshow('Square Interaction with Mouse Control', image)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()