import cv2
import mediapipe as mp
import pyautogui
import math
import time  # Added time module for tracking click delays

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

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

click_threshold = 0.04  # Adjust this threshold as needed
click_delay = 0.5  # Delay between clicks in seconds

# Track the last click times
last_left_click_time = 0
last_right_click_time = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip and convert color
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image
    result = hands.process(rgb_image)

    # Draw hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the index finger tip and thumb tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

            # Calculate the distance between the index finger tip and thumb tip
            distance_index = calculate_distance(index_finger_tip, thumb_tip)
            distance_middle = calculate_distance(middle_finger_tip, middle)

            # Print the distance for debugging
            print(f"Distance Index: {distance_index:.3f}, Distance Middle: {distance_middle:.3f}")

            # Move the mouse cursor (no delay for movement)
            index_x = int(middle.x * screen_width)
            index_y = int(middle.y * screen_height)
            pyautogui.moveTo(index_x, index_y)

            # Get current time for click delay check
            current_time = time.time()

            # Check if the distance is below the threshold to perform a click
            if distance_index < click_threshold and (current_time - last_left_click_time) > click_delay:
                pyautogui.click()
                print("Click!")
                last_left_click_time = current_time
            
            elif distance_middle < click_threshold and (current_time - last_right_click_time) > click_delay:
                pyautogui.mouseDown(button='middle')
                print("Right Click!")
                last_right_click_time = current_time
            else:
                pyautogui.mouseUp(button='middle')

    # Display
    cv2.imshow('Hand Detection', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()