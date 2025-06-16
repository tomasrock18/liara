import cv2
import numpy as np

from .camera import CameraBase
from .detector import Detector  


def extract_background(cam: CameraBase) -> np.ndarray:
    frame = None
    try:
        while True:
            frame = cam.get_frame()
            cv2.imshow("Background Extractor", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cv2.destroyAllWindows()
        return frame


def extract_master_contour(cam: CameraBase, bg: np.ndarray) -> np.ndarray:
    window_name = "Contour Extractor"
    cv2.namedWindow(window_name)

    def nothing(*args):
        pass

    cv2.createTrackbar("Threshold", window_name, 50, 255, nothing)

    contour_result = None

    try:
        while True:
            frame = cam.get_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY))

            threshold_val = cv2.getTrackbarPos("Threshold", window_name)
            _, thresh = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            display = frame.copy()
            cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

            cv2.putText(display,
                        f"Detected contours: {len(contours)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  
                break

        if contours:
            contour_result = max(contours, key=cv2.contourArea)

        return contour_result

    finally:
        cv2.destroyWindow(window_name)


def start_detector_tuning(
    cam: CameraBase,
    calibration_matrix: list[list[float]],
    rt_matrix: list[list[float]],
    bg: np.ndarray,
    master_contour: np.ndarray,
    camera_height: float
):
    window_name = "Detector Tuning"
    cv2.namedWindow(window_name)

    def nothing(*args):
        pass

    cv2.createTrackbar("Threshold", window_name, 50, 255, nothing)
    cv2.createTrackbar("Similarity x1000", window_name, 20, 100, nothing)
    cv2.createTrackbar("Min Area", window_name, 1000, 10000, nothing)

    detector = Detector(
        cam=cam,
        calibration_matrix=calibration_matrix,
        bg=bg,
        master_contour=master_contour,
        rt_matrix=rt_matrix,
        height=camera_height
    )

    try:
        while True:
            threshold = cv2.getTrackbarPos("Threshold", window_name)
            similarity = cv2.getTrackbarPos("Similarity x1000", window_name) / 1000
            min_area = cv2.getTrackbarPos("Min Area", window_name)

            frame = cam.get_frame()
            detections = detector.detect_objects(
                threshold=threshold,
                minimal_similarity=similarity,
                min_area=min_area
            )

            for contour, world_coords in detections:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = f"({world_coords[0]:.1f}, {world_coords[1]:.1f}, {world_coords[2]:.1f})"
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, label, (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.putText(frame,
                        f"Found: {len(detections)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2)

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cv2.destroyWindow(window_name)

