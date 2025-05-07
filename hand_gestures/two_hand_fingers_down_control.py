import cv2
import mediapipe as mp
import pyautogui
import time
import keyboard

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

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Variables to track click states and control enable/disable
clicked_left = False
clicked_middle = False
clicked_right = False  # For right click tracking
clicked_scroll = False  # For scroll tracking
control_enabled = True

def toggle_control(event):
    global control_enabled
    control_enabled = not control_enabled
    print(f"Control {'enabled' if control_enabled else 'disabled'}")

# Register the key press event to toggle control
keyboard.on_press_key('n', toggle_control)

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

    # Flip and convert color
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image
    result = hands.process(rgb_image)

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

    # Variables to track our two hands
    hand1_landmarks = None  # The hand that controls the mouse (first detected hand)
    hand2_landmarks = None  # The hand that provides the condition (second detected hand)
    
    # Check if hands are detected and control is enabled
    if result.multi_hand_landmarks and control_enabled:
        # Display how many hands are detected
        num_hands = len(result.multi_hand_landmarks)
        cv2.putText(
            image, 
            f"Hands detected: {num_hands}", 
            (10, 70), 
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
                (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Only move mouse if index and middle fingers are together on second hand
            if fingers_together:
                # Get landmarks for gesture detection from first hand
                base_middle = hand1_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                middle_finger_tip = hand1_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                index_finger_tip = hand1_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_mcp = hand1_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                
                # Map to screen coordinates using middle finger base
                cursor_x = int(base_middle.x * screen_width)
                cursor_y = int(base_middle.y * screen_height)
                
                # Move the mouse cursor
                pyautogui.moveTo(cursor_x, cursor_y)
                
                # Indicate that we're moving the mouse
                cv2.putText(
                    image, 
                    "Moving mouse", 
                    (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), 
                    2
                )

               
                
                # Check for middle finger down gesture (left click)
                middle_finger_down = is_finger_down(
                    hand1_landmarks,
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                )

                # Compute the distance between thumb and pinky
                thumb_pinky_distance = ((hand1_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x - hand1_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x) ** 2 +
                                        (hand1_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y - hand1_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y) ** 2) ** 0.5
                
                # Use wrist to index fingertip distance as a normalizing factor
                wrist = hand1_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                index_tip = hand1_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                hand_size_reference = ((wrist.x - index_tip.x) ** 2 + (wrist.y - index_tip.y) ** 2) ** 0.5
                
                # Normalize the distance
                normalized_distance = thumb_pinky_distance / hand_size_reference if hand_size_reference > 0 else float('inf')
                
                # Define thresholds for scrolling
                scroll_up_threshold = 0.2  # Adjust based on testing
                scroll_down_threshold = 0.2  # Adjust based on testing
                
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
                    print("Scroll up by thumb and pinky close")
                elif normalized_distance > scroll_down_threshold and normalized_distance < 0.5:  # Thumb and pinky are far
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
                    print("Scroll down by thumb and pinky far")
                
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
                        print("Left click by middle finger down")
                        clicked_left = True
                else:
                    clicked_left = False
                
                # Check for index finger up gesture (middle click hold/release)
                index_finger_up = is_finger_up(
                    hand1_landmarks,
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
                        print("Hold middle click by index finger down")
                        clicked_middle = True
                else:
                    if clicked_middle:
                        pyautogui.mouseUp(button='middle')
                        print("Release middle click")
                        clicked_middle = False
                
                # Check for ring finger down gesture (right click hold/release) - MODIFIED FOR HOLD
                ring_finger_down = is_finger_down(
                    hand1_landmarks,
                    mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_MCP
                )
                
                if ring_finger_down:
                    if not clicked_right:
                        pyautogui.mouseDown(button='right')  # Hold right click
                        cv2.putText(
                            image, 
                            "Right click hold", 
                            (10, 270), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (255, 0, 0), 
                            2
                        )
                        print("Hold right click by ring finger down")
                        clicked_right = True
                else:
                    if clicked_right:
                        pyautogui.mouseUp(button='right')  # Release right click
                        print("Release right click")
                        clicked_right = False

    # Display the resulting image
    cv2.imshow('Two-Hand Mouse Control with Gestures', image)
    
    # Exit with ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

    #scroll up with A key
    if keyboard.is_pressed('a'):
        pyautogui.scroll(75)
        
# Clean up
cap.release()
cv2.destroyAllWindows()