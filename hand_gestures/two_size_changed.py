import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
# Configure for two hands detection
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=2,  # Need to detect two hands
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width to 1280 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height to 720 pixels

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

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
    # This value will be consistent regardless of distance from camera
    threshold = 0.13  # Adjust as needed based on testing
    
    return normalized_distance < threshold

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip and convert color
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image
    result = hands.process(rgb_image)

    # Variables to track our two hands
    hand1_landmarks = None  # The hand that controls the mouse (first detected hand)
    hand2_landmarks = None  # The hand that provides the condition (second detected hand)
    
    # Check if hands are detected
    if result.multi_hand_landmarks:
        # Display how many hands are detected
        num_hands = len(result.multi_hand_landmarks)
        cv2.putText(
            image, 
            f"Hands detected: {num_hands}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Draw all hands and identify them based on detection order
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Assign landmarks to our variables based on detection order
            if idx == 0:
                hand1_landmarks = hand_landmarks
            elif idx == 1:
                hand2_landmarks = hand_landmarks
        
        # Check if we detected both hands
        if hand1_landmarks and hand2_landmarks:
            # Check if index and middle fingers are together in the second hand
            fingers_together = are_fingers_together(
                hand2_landmarks, 
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP
            )
            
            # Display status on image
            cv2.putText(
                image, 
                f"Fingers together: {fingers_together}", 
                (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Only move mouse if index and middle fingers are together on second hand
            if fingers_together:
                # Get middle finger MCP joint (base) position from first hand
                middle_mcp = hand1_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                
                # Map to screen coordinates
                cursor_x = int(middle_mcp.x * screen_width)
                cursor_y = int(middle_mcp.y * screen_height)
                
                # Move the mouse cursor
                pyautogui.moveTo(cursor_x, cursor_y)
                
                # Indicate that we're moving the mouse
                cv2.putText(
                    image, 
                    "Moving mouse", 
                    (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), 
                    2
                )

    # Display the resulting image
    cv2.imshow('Two-Hand Mouse Control', image)
    
    # Exit with ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()