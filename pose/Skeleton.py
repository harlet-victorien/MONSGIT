import cv2
import mediapipe as mp
import numpy as np
import time


class SkeletonDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the skeleton detector with MediaPipe Pose solution
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.pose = None
        
    def __enter__(self):
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pose:
            self.pose.close()
            
    def detect_landmarks(self, image):
        """
        Detect pose landmarks in the given image
        
        Args:
            image: Input RGB image
            
        Returns:
            Pose landmarks result object
        """
        return self.pose.process(image)
        
    def draw_landmarks(self, image, results):
        """
        Draw the pose landmarks on the image
        
        Args:
            image: Input BGR image
            results: Pose landmarks result object
            
        Returns:
            Image with landmarks drawn
        """
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        return image


class SkeletonVisualization:
    def __init__(self, source=0):
        """
        Initialize visualization with camera source
        
        Args:
            source: Camera index or video file path
        """
        self.source = source
        self.cap = None
        
    def __enter__(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {self.source}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
    def run(self, detector):
        """
        Run the skeleton detection visualization
        
        Args:
            detector: SkeletonDetector instance
        """
        fps_start_time = time.time()
        frame_count = 0
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
                
            # Calculate FPS
            frame_count += 1
            if frame_count >= 10:
                fps = frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                frame_count = 0
            else:
                fps = 0
                
            # Convert to RGB for MediaPipe
            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect landmarks
            results = detector.detect_landmarks(frame_rgb)
            
            # Draw landmarks
            frame.flags.writeable = True
            frame = detector.draw_landmarks(frame, results)
            
            # Add FPS counter
            if fps > 0:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the resulting frame
            cv2.imshow('MediaPipe Pose', cv2.flip(frame, 1))
            
            # Exit on ESC key
            if cv2.waitKey(5) & 0xFF == 27:
                break


def main():
    # Create detector and visualizer
    with SkeletonDetector() as detector, SkeletonVisualization() as visualizer:
        visualizer.run(detector)


if __name__ == "__main__":
    main()