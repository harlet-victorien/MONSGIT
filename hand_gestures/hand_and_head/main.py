from gestureController import HandFaceTracker
import cv2

handFaceTracker = HandFaceTracker()

while handFaceTracker.cap.isOpened():
    image, hand_centers, hand_types, hand_landmarks_dict, face_center, head_size = handFaceTracker.image_setup()

    if image is None:
        print("No image captured. Exiting...")
        break

    if face_center:
        squares, square_size_left, square_size_right = handFaceTracker.zones_setup(face_center, head_size)

        if handFaceTracker.display_camera:
            image = handFaceTracker.draw_square_list(image, squares, square_size_left, square_size_right)

        image = handFaceTracker.detect_hand_in_square(image, hand_centers, hand_types, squares, square_size_right, square_size_left)
        image = handFaceTracker.draw_squares_and_status(image, squares, square_size_right, square_size_left)
        image = handFaceTracker.process_gesture_control(image, hand_landmarks_dict, squares, square_size_left)

    if handFaceTracker.display_camera:
        cv2.imshow('Square Zone Detection', image)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break

    '''Using class variables to act'''
    print("Current zone:", handFaceTracker.current_zone)
    print("Current zoom:", handFaceTracker.zoomed)
    '''----------------------------'''

handFaceTracker.cap.release()
cv2.destroyAllWindows()
print("Script terminated.")