import mediapipe as mp
import pyautogui
import math
import cv2
import keyboard
import threading
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.9, min_tracking_confidence=0.9)
mp_draw = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables to track click states and control enable/disable
clicked_left = False
clicked_middle = False
control_enabled = True

# Variables for cursor position tracking
current_x, current_y = screen_width/2, screen_height/2  # Current mouse position
target_x, target_y = screen_width/2, screen_height/2    # Target position from model
mouse_update_rate = 0.01  # Time between mouse position updates in seconds (100Hz)
transition_speed = 0.08   # How quickly to move toward target (0.01-0.1)
mouse_thread_running = True

def toggle_control(event):
    global control_enabled
    control_enabled = not control_enabled
    print(f"Control {'enabled' if control_enabled else 'disabled'}")

# Register the key press event to toggle control
keyboard.on_press_key('n', toggle_control)

# Function to continuously update mouse position in a separate thread
def update_mouse_position():
    global current_x, current_y, target_x, target_y
    
    while mouse_thread_running:
        # Calculate next position with smooth transition toward target
        current_x = current_x + (target_x - current_x) * transition_speed
        current_y = current_y + (target_y - current_y) * transition_speed
        
        # Move the mouse to the interpolated position
        pyautogui.moveTo(int(current_x), int(current_y))
        
        # Sleep to maintain consistent update rate
        time.sleep(mouse_update_rate)

# Start the mouse update thread
mouse_thread = threading.Thread(target=update_mouse_position)
mouse_thread.daemon = True
mouse_thread.start()

try:
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks and control_enabled:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                base_middle = hand_landmarks.landmark[9]
                index_finger = hand_landmarks.landmark[8]
                thumb = hand_landmarks.landmark[4]
                middle_finger = hand_landmarks.landmark[12]

                # Position image
                x = int(base_middle.x * w)
                y = int(base_middle.y * h)
                cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)

                # Update the target position for the cursor
                target_x = base_middle.x * screen_width
                target_y = base_middle.y * screen_height

                # Distance between thumb and index
                ix, iy = int(index_finger.x * w), int(index_finger.y * h)
                tx, ty = int(thumb.x * w), int(thumb.y * h)
                distance_index_thumb = math.hypot(tx - ix, ty - iy)

                # Gesture: Middle finger down → left click
                if middle_finger.y > base_middle.y:
                    if not clicked_left:
                        pyautogui.click()
                        print("Left click by middle finger down")
                        clicked_left = True
                else:
                    if clicked_left:
                        clicked_left = False

                # Gesture: Index finger up → hold middle click
                if index_finger.y > base_middle.y:
                    if not clicked_middle:
                        pyautogui.mouseDown(button='middle')
                        print("Hold middle click by index finger up")
                        clicked_middle = True
                else:
                    if clicked_middle:
                        pyautogui.mouseUp(button='middle')
                        print("Release middle click")
                        clicked_middle = False

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup when exiting
    mouse_thread_running = False
    mouse_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()