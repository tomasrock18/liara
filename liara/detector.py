import numpy
import cv2

from .camera import CameraBase


class Detector:
    def __init__(
            self,
            cam: CameraBase,
            calibration_matrix: list[list[float]],
            bg: numpy.ndarray,
            master_contour: numpy.ndarray,
            rt_matrix: list[list[float]],
    ):
        self.cam = cam
        self.bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        self.master_contour = master_contour
        self.K = numpy.array(calibration_matrix, dtype=numpy.float64)
        self.RT = numpy.array(rt_matrix, dtype=numpy.float64)
        self.K_inv = numpy.linalg.inv(self.K)

        self.R = self.RT[:, :3]
        self.T = self.RT[:, 3:]

    def detect_objects(self) -> list[tuple[numpy.ndarray, numpy.ndarray]]:
        result = []

        frame = self.cam.get_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, self.bg_gray)

        _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
        kernel = numpy.ones((5, 5), numpy.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            similarity = cv2.matchShapes(self.master_contour, cnt, cv2.CONTOURS_MATCH_I1, 0.0)
            if similarity < 0.1:
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Пиксель → нормализованная координата в камере
                pixel_point = numpy.array([[cx], [cy], [1]], dtype=numpy.float64)
                cam_point = self.K_inv @ pixel_point

                # Подставим Z=1 (глубину по умолчанию)
                cam_point_homogeneous = numpy.vstack([cam_point[:2], [1]])

                # Перевод в мировые координаты
                world_point = numpy.linalg.inv(self.R) @ (cam_point_homogeneous - self.T)

                result.append((cnt, world_point.flatten()))

        return result
