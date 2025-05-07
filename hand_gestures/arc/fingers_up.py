import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Delay between clicks in seconds
click_delay = 0.5

# Track the last click time
last_click_time = 0

# Function to determine if a finger is up
def is_finger_up(landmarks, finger_tip_id, finger_mcp_id):
    # Check if tip of finger is higher than MCP joint (lower y value)
    return landmarks.landmark[finger_tip_id].y + 0.1 < landmarks.landmark[finger_mcp_id].y

def count_fingers_up(landmarks):
    finger_count = 0
            
    # Check index finger
    if is_finger_up(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                   mp_hands.HandLandmark.INDEX_FINGER_MCP):
        finger_count += 1
        
    # Check middle finger
    if is_finger_up(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                   mp_hands.HandLandmark.MIDDLE_FINGER_MCP):
        finger_count += 1
        
    # Check ring finger
    if is_finger_up(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, 
                   mp_hands.HandLandmark.RING_FINGER_MCP):
        finger_count += 1
        
    # Check pinky finger
    if is_finger_up(landmarks, mp_hands.HandLandmark.PINKY_TIP, 
                   mp_hands.HandLandmark.PINKY_MCP):
        finger_count += 1
        
    return finger_count

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip and convert color
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image
    result = hands.process(rgb_image)

    # Draw hand landmarks and handle mouse control
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Count fingers that are up
            fingers_up = count_fingers_up(hand_landmarks)
            
            # Display finger count on image
            cv2.putText(image, f"Fingers: {fingers_up}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Use wrist position for cursor movement
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            cursor_x = int(wrist.x * screen_width)
            cursor_y = int(wrist.y * screen_height)
            pyautogui.moveTo(cursor_x, cursor_y)
            
            # Get current time for click delay check
            current_time = time.time()
            
            # Only perform clicks if enough time has passed since last click
            if (current_time - last_click_time) > click_delay:
                if fingers_up == 1:
                    pyautogui.click()  # Left click
                    print("Left Click!")
                    last_click_time = current_time
                elif fingers_up == 2:
                    pyautogui.click(button='right')  # Right click
                    print("Right Click!")  
                    last_click_time = current_time
                elif fingers_up == 3:
                    pyautogui.click(button='middle')  # Middle click
                    print("Middle Click!")
                    last_click_time = current_time

    # Display the resulting image
    cv2.imshow('Finger Count Mouse Control', image)
    
    # Exit with ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()