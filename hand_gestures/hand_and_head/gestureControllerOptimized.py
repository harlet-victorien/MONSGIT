import cv2
import mediapipe as mp
import numpy as np
import keyboard

class HandFaceTracker:
    """
    A class to track hand and face landmarks using MediaPipe, and perform gesture-based control.
    """

    def __init__(self):
        """
        Initialize MediaPipe hands and face mesh, webcam, and control variables.
        """
        # Initialize MediaPipe hands and face mesh
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize the solutions
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000)

        # Variables to track control enable/disable
        self.control_enabled = True
        self.display_camera = True
        self.zoomed = False
        self.last_pinky_thumb_state = False

        # Register the key press event to toggle control
        keyboard.on_press_key('n', self.toggle_control)

        # Register the key press event to toggle display
        keyboard.on_press_key('d', self.toggle_display)

    def toggle_control(self, event):
        """Toggle the control state between enabled and disabled."""
        self.control_enabled = not self.control_enabled
        print(f"Control {'enabled' if self.control_enabled else 'disabled'}")

    def toggle_display(self, event):
        """Toggle the camera display state between enabled and disabled."""
        self.display_camera = not self.display_camera
        print(f"Camera display {'enabled' if self.display_camera else 'disabled'}")

    def calculate_distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def is_point_in_rect(self, point, rect_x, rect_y, rect_w, rect_h):
        """Check if a point is inside a rectangle."""
        x, y = point
        return rect_x <= x <= rect_x + rect_w and rect_y <= y <= rect_y + rect_h

    def is_point_in_diagonal_zone(self, point, square_x, square_y, square_size, zone_number):
        """Check if a point is inside a diagonal zone of a square."""
        x, y = point
        rel_x = (x - square_x) / square_size
        rel_y = (y - square_y) / square_size

        if 0 <= rel_x <= 1 and 0 <= rel_y <= 1:
            if zone_number == 0:  # Top zone
                return rel_y < 0.5 and rel_y < rel_x and rel_y < 1 - rel_x
            elif zone_number == 1:  # Right zone
                return rel_x > 0.5 and rel_y < rel_x and rel_y > 1 - rel_x
            elif zone_number == 2:  # Bottom zone
                return rel_y > 0.5 and rel_y > rel_x and rel_y > 1 - rel_x
            elif zone_number == 3:  # Left zone
                return rel_x < 0.5 and rel_y > rel_x and rel_y < 1 - rel_x
        return False

    def are_fingers_together(self, landmarks, finger1_tip_id, finger2_tip_id):
        """Check if two fingers are together based on normalized distance."""
        tip1 = landmarks.landmark[finger1_tip_id]
        tip2 = landmarks.landmark[finger2_tip_id]
        tips_distance = ((tip1.x - tip2.x) ** 2 + (tip1.y - tip2.y) ** 2) ** 0.5

        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        hand_size_reference = ((wrist.x - index_tip.x) ** 2 + (wrist.y - index_tip.y) ** 2) ** 0.5

        normalized_distance = tips_distance / hand_size_reference if hand_size_reference > 0 else float('inf')
        threshold = 0.13
        return normalized_distance < threshold

    def draw_square_with_zones(self, image, x, y, square_size, name):
        """Draw a square with diagonal zones."""
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + square_size, y + square_size), (0, 0, 255), -1)
        alpha = 0.2
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        if name == 'Left Zone':
            cv2.line(image, (x, y), (x + square_size, y + square_size), (255, 255, 255), 2)
            cv2.line(image, (x + square_size, y), (x, y + square_size), (255, 255, 255), 2)

            zone_names = ["Zone 1", "Zone 2", "Zone 3", "Zone 4"]
            zone_positions = [
                (x + square_size//2, y + square_size//4 - 10),
                (x + 3*square_size//4 + 10, y + square_size//2),
                (x + square_size//2, y + 3*square_size//4 + 10),
                (x + square_size//4 - 10, y + square_size//2)
            ]

            for zone_pos, zone_name in zip(zone_positions, zone_names):
                cv2.putText(image, zone_name, zone_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image

    def run(self):
        """Run the main loop to process video frames and perform gesture-based control."""
        print("Script is running. Press 'n' to toggle control, 'd' to toggle camera display, ESC to exit.")

        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                break

            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            hand_results = self.hands.process(rgb_image)
            face_results = self.face_mesh.process(rgb_image)

            face_center = None
            head_size = 0
            hand_centers = []
            hand_types = []
            hand_landmarks_dict = {}
            status_text_main = "Control: DISABLED"

            if hand_results.multi_hand_landmarks:
                num_hands = len(hand_results.multi_hand_landmarks)
                print(f"Hands detected: {num_hands}", end="\r")

                for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    handedness = hand_results.multi_handedness[idx].classification[0].label
                    hand_type = "Left" if handedness == "Right" else "Right"
                    hand_types.append(hand_type)
                    hand_landmarks_dict[hand_type] = hand_landmarks

                    ih, iw, _ = image.shape
                    wrist = hand_landmarks.landmark[0]
                    wrist_pos = (int(wrist.x * iw), int(wrist.y * ih))
                    hand_centers.append(wrist_pos)

                    index_finger_tip = hand_landmarks.landmark[8]
                    index_pos = (int(index_finger_tip.x * iw), int(index_finger_tip.y * ih))
                    hand_centers.append(index_pos)
                    hand_types.append(f"{hand_type}_tip")

                    if self.display_camera:
                        self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        cv2.circle(image, wrist_pos, 5, (0, 255, 0), -1)
                        cv2.circle(image, index_pos, 5, (255, 255, 0), -1)

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    ih, iw, _ = image.shape
                    nose_tip = face_landmarks.landmark[4]
                    face_center = (int(nose_tip.x * iw), int(nose_tip.y * ih))

                    if self.display_camera:
                        cv2.circle(image, face_center, 8, (0, 0, 255), -1)

                    left_eye = face_landmarks.landmark[33]
                    right_eye = face_landmarks.landmark[263]
                    left_eye_pos = (int(left_eye.x * iw), int(left_eye.y * ih))
                    right_eye_pos = (int(right_eye.x * iw), int(right_eye.y * ih))
                    head_size = self.calculate_distance(left_eye_pos, right_eye_pos)

                    square_size = int(head_size * 4)
                    distance_factor = 1.5

                    squares = [
                        {
                            'x': int(face_center[0] - distance_factor * head_size - square_size),
                            'y': int(face_center[1] - square_size/2),
                            'name': 'Right Zone',
                            'hand_detected': None
                        },
                        {
                            'x': int(face_center[0] + distance_factor * head_size),
                            'y': int(face_center[1] - square_size/2),
                            'name': 'Left Zone',
                            'hand_detected': None
                        }
                    ]

                    if self.display_camera:
                        for square in squares:
                            x, y = square['x'], square['y']
                            image = self.draw_square_with_zones(image, x, y, square_size, square['name'])

                    if len(hand_centers) >= 2:
                        for i in range(0, len(hand_centers), 2):
                            if i+1 < len(hand_centers):
                                wrist_idx = i
                                index_idx = i+1

                                for square in squares:
                                    x, y = square['x'], square['y']
                                    wrist_in_square = self.is_point_in_rect(hand_centers[wrist_idx], x, y, square_size, square_size)
                                    index_in_square = self.is_point_in_rect(hand_centers[index_idx], x, y, square_size, square_size)

                                    if wrist_in_square and index_in_square:
                                        if self.display_camera:
                                            overlay = image.copy()
                                            cv2.rectangle(overlay, (x, y), (x + square_size, y + square_size), (0, 255, 0), -1)
                                            image = cv2.addWeighted(overlay, 0.2, image, 1 - 0.2, 0)

                                        base_hand_type = hand_types[wrist_idx]
                                        square['hand_detected'] = base_hand_type

                    if self.display_camera:
                        for square in squares:
                            x, y = square['x'], square['y']
                            color = (0, 255, 0) if square['hand_detected'] else (0, 0, 255)
                            cv2.rectangle(image, (x, y), (x + square_size, y + square_size), color, 2)
                            status_text = f"{square['name']}: {square['hand_detected']}" if square['hand_detected'] else f"{square['name']}"
                            cv2.putText(image, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    left_hand_in_left_square = False
                    right_hand_in_right_square = False

                    for square in squares:
                        if square['name'] == 'Left Zone' and square['hand_detected'] == 'Left':
                            left_hand_in_left_square = True
                        elif square['name'] == 'Right Zone' and square['hand_detected'] == 'Right':
                            right_hand_in_right_square = True

                    if self.control_enabled and left_hand_in_left_square and right_hand_in_right_square:
                        if 'Left' in hand_landmarks_dict and 'Right' in hand_landmarks_dict:
                            left_hand = hand_landmarks_dict['Left']
                            right_hand = hand_landmarks_dict['Right']

                            fingers_together = self.are_fingers_together(
                                right_hand,
                                self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                            )

                            if fingers_together:
                                status_text_main = "Control: ENABLED"

                                pinky_thumb_together = self.are_fingers_together(
                                    left_hand,
                                    self.mp_hands.HandLandmark.PINKY_TIP,
                                    self.mp_hands.HandLandmark.THUMB_TIP
                                )

                                if pinky_thumb_together and not self.last_pinky_thumb_state:
                                    self.zoomed = not self.zoomed
                                    print(f"ZOOM STATE TOGGLED: {'ZOOMED IN' if self.zoomed else 'ZOOMED OUT'}")

                                self.last_pinky_thumb_state = pinky_thumb_together

                                if self.display_camera:
                                    zoom_status = "ZOOMED IN" if self.zoomed else "ZOOMED OUT"
                                    cv2.putText(image, f"Zoom: {zoom_status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                    pinky_status = "Pinky-Thumb Together" if pinky_thumb_together else "Pinky-Thumb Apart"
                                    cv2.putText(image, pinky_status, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                                left_zone_square = next((sq for sq in squares if sq['name'] == 'Left Zone'), None)
                                if left_zone_square and 'Left' in hand_landmarks_dict:
                                    index_tip = hand_landmarks_dict['Left'].landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                                    ih, iw, _ = image.shape
                                    index_tip_pos = (int(index_tip.x * iw), int(index_tip.y * ih))

                                    if self.display_camera:
                                        cv2.circle(image, index_tip_pos, 8, (0, 255, 255), -1)

                                    sq_x = left_zone_square['x']
                                    sq_y = left_zone_square['y']

                                    current_zone = None
                                    for i in range(4):
                                        if self.is_point_in_diagonal_zone(index_tip_pos, sq_x, sq_y, square_size, i):
                                            current_zone = f"Zone {i+1}"
                                            break

                                    if self.display_camera and current_zone:
                                        cv2.putText(image, current_zone, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            if self.display_camera:
                cv2.putText(image, status_text_main, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Square Zone Detection', image)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("Script terminated.")

# Example usage
if __name__ == "__main__":
    tracker = HandFaceTracker()
    tracker.run()
