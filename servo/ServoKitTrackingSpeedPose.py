import sys
from ultralytics import YOLO
import numpy as np
from jtop import jtop
import cv2
import time
import ServoKitCalib

np.bool = np.bool_  # Compatibilité avec les anciennes versions de numpy

class PoseTracker:
    """
    Classe de gestion d'une caméra avec ServoKit pour le suivi pan/tilt,
    avec possibilité d'intégrer un modèle de détection ultérieurement.
    """

    def __init__(self, fps=30, width=1920, height=1080):
        """
        Initialise la caméra, les paramètres de tracking et les servos.

        Args:
            fps (int): Nombre d'images par seconde.
            width (int): Largeur des frames vidéo.
            height (int): Hauteur des frames vidéo.
        """
        self.fps = fps
        self.frame_rate = 1 / fps
        self.v_max_pan = 90
        self.v_max_tilt = 90
        self.dead_zone_x = 0.1
        self.dead_zone_y = 0.1
        self.width = width
        self.height = height

        self.servo_kit = ServoKitCalib.ServoKit(num_ports = 4)
        self.cam_source = self.servo_kit.NumCamera
        self.servo_kit.setAngleDeg(0, 0)
        self.servo_kit.setAngleDeg(1, 0)

        self.camera = cv2.VideoCapture(self._gst_pipeline(), cv2.CAP_GSTREAMER)

        self.new_angle_pan = 0
        self.new_angle_tilt = 0

        self.model = None  # Modèle non chargé par défaut
        self.last_time = time.time()

    def _gst_pipeline(self):
        """
        Construit le pipeline GStreamer pour capturer la vidéo.

        Returns:
            str: pipeline GStreamer.
        """
        return (
            f"v4l2src device=/dev/video{self.cam_source} ! "
            f"image/jpeg, width={self.width}, height={self.height}, framerate={self.fps}/1 ! "
            f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink"
        )

    def load_model(self, model_path):
        """
        Charge un modèle YOLO optimisé (ex: TensorRT).

        Args:
            model_path (str): Chemin vers le fichier du modèle YOLO.
        """
        self.model = YOLO(model_path)
        print(f"Modèle YOLO chargé depuis : {model_path}")

    def ptz_tracking(self, angle_pan, angle_tilt, x, y):
        """
        Calcule les nouveaux angles PAN/TILT à partir d'une position d'objet.

        Args:
            angle_pan (float): Angle actuel PAN.
            angle_tilt (float): Angle actuel TILT.
            x (int): Position X du point.
            y (int): Position Y du point.

        Returns:
            tuple: Nouveaux angles (pan, tilt).
        """
        centre_x = self.width / 2
        centre_y = self.height / 2

        offset_x = -(x - centre_x) / (centre_x / 1)
        offset_y = -(y - centre_y) / (centre_y / 1)

        speed_pan = speed_tilt = 0

        if abs(offset_x) > self.dead_zone_x:
            speed_pan = self.v_max_pan * offset_x * self.frame_rate

        if abs(offset_y) > self.dead_zone_y:
            speed_tilt = self.v_max_tilt * offset_y * self.frame_rate

        angle_pan += speed_pan
        angle_tilt += speed_tilt

        return angle_pan, angle_tilt

    def run_with_model(self):
        """
        Lance la boucle principale avec le modèle YOLO chargé.
        """
        if self.model is None:
            raise RuntimeError("Modèle YOLO non chargé. Utilisez `load_model()` d'abord.")

        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Erreur de lecture caméra")
                break

            frame = cv2.flip(frame, 0)
            self.height, self.width, _ = frame.shape

            if cv2.waitKey(1) == ord("q"):
                break

            results = self.model([frame], stream=True, batch=1, verbose=False)
            for r in results:
                keypoints = r.keypoints.xy

                for j in range(min(len(keypoints), 2)):
                    if len(keypoints[0]) > 0:
                        x, y = int(keypoints[j][0][0]), int(keypoints[j][0][1])
                        cv2.circle(frame, (x, y), 5, (255, 255, 0), -1)

                        self.new_angle_pan, self.new_angle_tilt = self.ptz_tracking(
                            self.new_angle_pan, self.new_angle_tilt, x, y
                        )

                        self.servo_kit.setAngleDeg(0, self.new_angle_pan)
                        self.servo_kit.setAngleDeg(1, self.new_angle_tilt)

            cv2.imshow('cam 1', frame)

        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = PoseTracker()
    print(cv2.getBuildInformation())

    # Charger le modèle manuellement après instanciation
    tracker.load_model("../model/Yolo/yolo11n-pose.engine")
    tracker.run_with_model()
