import cv2
import mediapipe as mp
import numpy as np
import keyboard

class GestureController:
    """
    A class for controlling interfaces using hand gestures and head tracking.
    
    This controller uses a webcam to track hands and face position,
    detect specific gestures, and trigger actions based on hand positions
    relative to virtual squares around the head.
    """
    
    def __init__(self):
        """Initialize the GestureController with MediaPipe solutions and webcam setup."""
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

        # Control state variables
        self.control_enabled = True
        self.display_camera = True
        self.zoomed = False
        self.last_pinky_thumb_state = False
        
        # Register keyboard event handlers
        keyboard.on_press_key('n', self.toggle_control)
        keyboard.on_press_key('d', self.toggle_display)

    def toggle_control(self, event):
        """Toggle the control mode on/off."""
        self.control_enabled = not self.control_enabled
        print(f"Control {'enabled' if self.control_enabled else 'disabled'}")

    def toggle_display(self, event):
        """Toggle the camera display on/off."""
        self.display_camera = not self.display_camera
        print(f"Camera display {'enabled' if self.display_camera else 'disabled'}")

    def calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two 2D points.
        
        Args:
            point1 (tuple): First point coordinates (x, y)
            point2 (tuple): Second point coordinates (x, y)
            
        Returns:
            float: Distance between the points
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def is_point_in_rect(self, point, rect_x, rect_y, rect_w, rect_h):
        """
        Check if a point is inside a rectangle.
        
        Args:
            point (tuple): Point coordinates (x, y)
            rect_x (int): Rectangle left x coordinate
            rect_y (int): Rectangle top y coordinate
            rect_w (int): Rectangle width
            rect_h (int): Rectangle height
            
        Returns:
            bool: True if point is in rectangle, False otherwise
        """
        x, y = point
        return rect_x <= x <= rect_x + rect_w and rect_y <= y <= rect_y + rect_h

    def is_point_in_diagonal_zone(self, point, square_x, square_y, square_size, zone_number):
        """
        Check if a point is inside a specific diagonal zone of a square.
        
        Args:
            point (tuple): Point coordinates (x, y)
            square_x (int): Square left x coordinate
            square_y (int): Square top y coordinate
            square_size (int): Size of the square
            zone_number (int): Zone index (0-3) representing top, right, bottom, left
            
        Returns:
            bool: True if point is in the specified zone, False otherwise
        """
        x, y = point
        
        # Normalize coordinates to be relative to the square (0-1)
        rel_x = (x - square_x) / square_size
        rel_y = (y - square_y) / square_size
        
        # Check which diagonal zone the point belongs to using diagonal divisions
        if 0 <= rel_x <= 1 and 0 <= rel_y <= 1:
            if zone_number == 0:  # Top zone (above both diagonals)
                return rel_y < 0.5 and rel_y < rel_x and rel_y < 1 - rel_x
            elif zone_number == 1:  # Right zone (right of both diagonals)
                return rel_x > 0.5 and rel_y < rel_x and rel_y > 1 - rel_x
            elif zone_number == 2:  # Bottom zone (below both diagonals)
                return rel_y > 0.5 and rel_y > rel_x and rel_y > 1 - rel_x
            elif zone_number == 3:  # Left zone (left of both diagonals)
                return rel_x < 0.5 and rel_y > rel_x and rel_y < 1 - rel_x
        
        return False

    def are_fingers_together(self, landmarks, finger1_tip_id, finger2_tip_id):
        """
        Check if two fingers are close together using normalized distance.
        
        Args:
            landmarks: MediaPipe hand landmarks
            finger1_tip_id (int): Landmark ID for first finger tip
            finger2_tip_id (int): Landmark ID for second finger tip
            
        Returns:
            bool: True if fingers are together, False otherwise
        """
        tip1 = landmarks.landmark[finger1_tip_id]
        tip2 = landmarks.landmark[finger2_tip_id]
        
        # Calculate Euclidean distance in 2D space (x and y) between the two fingertips
        tips_distance = ((tip1.x - tip2.x) ** 2 + (tip1.y - tip2.y) ** 2) ** 0.5
        
        # Use wrist to index fingertip distance as a normalizing factor
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Calculate hand size reference (wrist to index tip distance)
        hand_size_reference = ((wrist.x - index_tip.x) ** 2 + (wrist.y - index_tip.y) ** 2) ** 0.5
        
        # Normalize the distance by dividing by the hand size reference
        normalized_distance = tips_distance / hand_size_reference if hand_size_reference > 0 else float('inf')
        
        # Define a threshold for normalized "togetherness"
        threshold = 0.13  # Adjust as needed based on testing
        
        return normalized_distance < threshold
    
    def process_frame(self, image):
        """
        Process a single camera frame for hand and face detection.
        
        Args:
            image: OpenCV image frame
            
        Returns:
            image: Processed image with visualizations (if display enabled)
        """
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process for hands and face
        hand_results = self.hands.process(rgb_image)
        face_results = self.face_mesh.process(rgb_image)
        
        # Variables to store important points
        face_center = None
        head_size = 0
        hand_centers = []
        hand_types = []
        hand_landmarks_dict = {}  # To store hand landmarks for each hand type
        
        # Process hand landmarks
        if hand_results.multi_hand_landmarks:
            # Print how many hands are detected
            num_hands = len(hand_results.multi_hand_landmarks)
            print(f"Hands detected: {num_hands}", end="\r")
            
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Determine if it's left or right hand
                handedness = hand_results.multi_handedness[idx].classification[0].label
                hand_type = "Left" if handedness == "Right" else "Right"  # Flipped due to mirror view
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
                
                if self.display_camera:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Mark wrist center
                    cv2.circle(image, wrist_pos, 5, (0, 255, 0), -1)
                    cv2.circle(image, index_pos, 5, (255, 255, 0), -1)  # Yellow for finger tip
        
        # Handle face mesh landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Get important face landmarks
                ih, iw, _ = image.shape
                
                # Use nose tip as face center (landmark 4)
                nose_tip = face_landmarks.landmark[4]
                face_center = (int(nose_tip.x * iw), int(nose_tip.y * ih))
                
                if self.display_camera:
                    # Draw nose point with larger circle for better visibility
                    cv2.circle(image, face_center, 8, (0, 0, 255), -1)
                
                # Calculate head size (using distance between eyes)
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                left_eye_pos = (int(left_eye.x * iw), int(left_eye.y * ih))
                right_eye_pos = (int(right_eye.x * iw), int(right_eye.y * ih))
                head_size = self.calculate_distance(left_eye_pos, right_eye_pos)
                
                # Define squares at normalized positions around the head
                square_size = int(head_size * 4)  # Size of squares relative to head size
                distance_factor = 1.5  # Distance from head center as a factor of head_size
                
                squares = [
                    # Left square (physically on left, but Right Zone)
                    {
                        'x': int(face_center[0] - distance_factor * head_size - square_size),
                        'y': int(face_center[1] - square_size/2),
                        'name': 'Right Zone',
                        'hand_detected': None
                    },
                    # Right square (physically on right, but Left Zone)
                    {
                        'x': int(face_center[0] + distance_factor * head_size),
                        'y': int(face_center[1] - square_size/2),
                        'name': 'Left Zone',
                        'hand_detected': None
                    }
                ]
                
                self.process_squares(image, squares, hand_centers, hand_types, square_size)
                self.process_gestures(image, squares, hand_landmarks_dict, face_center)
        
        # Display the camera feed only if display_camera is True
        if self.display_camera:
            # Display status information
            status_text = "Control: ENABLED" if self.control_enabled else "Control: DISABLED"
            cv2.putText(
                image, 
                status_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
        
        return image
            
    def process_squares(self, image, squares, hand_centers, hand_types, square_size):
        """
        Process the detection squares and check for hands inside them.
        
        Args:
            image: OpenCV image frame
            squares (list): List of square dictionaries
            hand_centers (list): List of hand center positions
            hand_types (list): List of hand type labels
            square_size (int): Size of the squares
        """
        if self.display_camera:
            # Draw the squares and check for hands inside them
            for square in squares:
                x, y = square['x'], square['y']
                name = square['name']
                
                # Initialize with inactive color (red)
                color = (0, 0, 255)  # Red by default
                
                # Make squares more visible with semi-transparent fill
                overlay = image.copy()
                cv2.rectangle(overlay, (x, y), (x + square_size, y + square_size), (0, 0, 255), -1)
                alpha = 0.2  # Transparency factor
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                
                # If this is the "LEFT ZONE" (right square in our setup), draw the diagonal divisions
                if name == 'Left Zone':
                    # Draw diagonal division lines
                    cv2.line(image, (x, y), (x + square_size, y + square_size), (255, 255, 255), 2)
                    cv2.line(image, (x + square_size, y), (x, y + square_size), (255, 255, 255), 2)
                    
                    # Add zone labels
                    zone_names = ["Zone 1", "Zone 2", "Zone 3", "Zone 4"]
                    zone_positions = [
                        (x + square_size//2, y + square_size//4),  # Top
                        (x + 3*square_size//4, y + square_size//2),  # Right
                        (x + square_size//2, y + 3*square_size//4),  # Bottom
                        (x + square_size//4, y + square_size//2)  # Left
                    ]
                    
                    for i, (zone_pos, zone_name) in enumerate(zip(zone_positions, zone_names)):
                        cv2.putText(
                            image,
                            zone_name,
                            zone_pos,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1
                        )
        
        # Check for hands where both wrist and index finger are inside the square
        if len(hand_centers) >= 2:  # Make sure we have at least some hand points
            for i in range(0, len(hand_centers), 2):  # Process pairs of points (wrist and index)
                if i+1 < len(hand_centers):  # Make sure we have both wrist and index
                    wrist_idx = i
                    index_idx = i+1
                    
                    for square in squares:
                        x, y = square['x'], square['y']
                        
                        # Check if both points are in the square
                        wrist_in_square = self.is_point_in_rect(hand_centers[wrist_idx], x, y, square_size, square_size)
                        index_in_square = self.is_point_in_rect(hand_centers[index_idx], x, y, square_size, square_size)
                        
                        # Only consider hand in square if both wrist and index finger are inside
                        if wrist_in_square and index_in_square:
                            # Get the hand type (without "_tip" suffix)
                            base_hand_type = hand_types[wrist_idx]
                            
                            # Store which hand is in this square
                            square['hand_detected'] = base_hand_type
                            
                            if self.display_camera:
                                # Make square green with semi-transparent fill when hand is inside
                                overlay = image.copy()
                                cv2.rectangle(overlay, (x, y), (x + square_size, y + square_size), (0, 255, 0), -1)
                                alpha = 0.2  # Transparency factor
                                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        if self.display_camera:
            # Finalize drawing squares
            for square in squares:
                x, y = square['x'], square['y']
                name = square['name']
                which_hand = square['hand_detected'] if square['hand_detected'] else ""
                
                # Determine the color based on hand detection
                color = (0, 255, 0) if square['hand_detected'] else (0, 0, 255)
                
                # Draw the square outline
                cv2.rectangle(image, (x, y), (x + square_size, y + square_size), color, 2)
                
                # Display zone name and hand detection status
                status_text = f"{name}: {which_hand}" if which_hand else f"{name}"
                cv2.putText(image, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, color, 2)

    def process_gestures(self, image, squares, hand_landmarks_dict, face_center):
        """
        Process gestures and determine actions based on hand positions and poses.
        
        Args:
            image: OpenCV image frame
            squares (list): List of square dictionaries
            hand_landmarks_dict (dict): Dictionary of hand landmarks by type
            face_center (tuple): Center point of detected face
        """
        # Check if both hands are in their respective squares
        left_hand_in_left_square = False
        right_hand_in_right_square = False
        
        for square in squares:
            if square['name'] == 'Left Zone' and square['hand_detected'] == 'Left':
                left_hand_in_left_square = True
            elif square['name'] == 'Right Zone' and square['hand_detected'] == 'Right':
                right_hand_in_right_square = True
        
        # Only detect diagonal zones if control is enabled and both hands are in position
        if self.control_enabled and left_hand_in_left_square and right_hand_in_right_square:
            if 'Left' in hand_landmarks_dict and 'Right' in hand_landmarks_dict:
                # Get hand landmarks for gesture detection
                left_hand = hand_landmarks_dict['Left']
                right_hand = hand_landmarks_dict['Right']
                
                # Check if index and middle fingers are together on the right hand
                fingers_together = self.are_fingers_together(
                    right_hand, 
                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                )
                
                print("CONTROL ACTIVE - Both hands in position")
                print(f"Fingers together: {'YES' if fingers_together else 'NO'}")
                
                # Only detect zones and check for zoom toggle if fingers are together
                if fingers_together:
                    # Check if pinky and thumb are together on the LEFT hand (action hand)
                    pinky_thumb_together = self.are_fingers_together(
                        left_hand,
                        self.mp_hands.HandLandmark.PINKY_TIP,
                        self.mp_hands.HandLandmark.THUMB_TIP
                    )
                    
                    # Toggle zoomed state only on change from not-together to together
                    if pinky_thumb_together and not self.last_pinky_thumb_state:
                        self.zoomed = not self.zoomed
                        print(f"ZOOM STATE TOGGLED: {'ZOOMED IN' if self.zoomed else 'ZOOMED OUT'}")
                    
                    # Update the previous state
                    self.last_pinky_thumb_state = pinky_thumb_together
                    
                    # Display zoom status if camera display is enabled
                    if self.display_camera:
                        zoom_status = "ZOOMED IN" if self.zoomed else "ZOOMED OUT"
                        cv2.putText(
                            image,
                            f"Zoom: {zoom_status}",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            2
                        )
                        
                        # Visualize pinky-thumb state
                        pinky_status = "Pinky-Thumb Together" if pinky_thumb_together else "Pinky-Thumb Apart"
                        cv2.putText(
                            image,
                            pinky_status,
                            (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )
                    
                    # Find the Left Zone square (which is physically on the right) for zone detection
                    left_zone_square = None
                    for sq in squares:
                        if sq['name'] == 'Left Zone':
                            left_zone_square = sq
                            break
                    
                    # Get left hand's index finger tip position for zone detection
                    if left_zone_square and 'Left' in hand_landmarks_dict:
                        index_tip = hand_landmarks_dict['Left'].landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        ih, iw, _ = image.shape
                        index_tip_pos = (int(index_tip.x * iw), int(index_tip.y * ih))
                        
                        if self.display_camera:
                            # Mark index finger tip with distinct circle for visibility
                            cv2.circle(image, index_tip_pos, 8, (0, 255, 255), -1)
                        
                        # Get square boundaries
                        sq_x = left_zone_square['x']
                        sq_y = left_zone_square['y']
                        
                        # Check which diagonal zone the index finger tip is in
                        current_zone = None
                        
                        for i in range(4):
                            if self.is_point_in_diagonal_zone(index_tip_pos, sq_x, sq_y, square_size, i):
                                current_zone = f"Zone {i+1}"
                                # Print zone to console
                                print(f"Index finger tip in: {current_zone}")
                                break
                        
                        if current_zone is None and self.is_point_in_rect(index_tip_pos, sq_x, sq_y, square_size, square_size):
                            # This handles edge cases where the finger is in the square but not clearly in a zone
                            print("Index finger tip is in square but zone detection is unclear")

                        # Draw the zone name on the image with larger text and more visible position
                        if self.display_camera and current_zone:
                            cv2.putText(
                                image,
                                f"Active: {current_zone}",
                                (10, 160),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0, 0, 255),  # Red text for better visibility
                                2
                            )
    
    def run(self):
        """Run the main gesture control loop."""
        print("Script is running. Press 'n' to toggle control, 'd' to toggle camera display, ESC to exit.")
        
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                break
                
            # Flip horizontally for a selfie-view
            image = cv2.flip(image, 1)
            
            # Process the frame
            processed_image = self.process_frame(image)
            
            # Display the camera feed if enabled
            if self.display_camera:
                cv2.imshow('Square Zone Detection', processed_image)
            
            # Check for keyboard input even when not displaying the camera
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Script terminated.")


# Create and run the controller when script is executed
if __name__ == "__main__":
    controller = GestureController()
    controller.run()