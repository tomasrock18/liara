import numpy as np
import cv2


class Detector:
    def __init__(
            self,
            cam,
            calibration_matrix: list[list[float]],
            rt_matrix: list[list[float]],
            bg: np.ndarray,
            master_contour: np.ndarray,
            height: float
    ):
        self.cam = cam
        self.bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        self.master_contour = master_contour
        self.K = np.array(calibration_matrix, dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)
        self.camera_height = height
        self.rt_matrix = np.array(rt_matrix, dtype=np.float64)

    def detect_objects(
            self,
            threshold=50,
            minimal_similarity=0.02,
            min_area=1000,
            max_aspect_ratio=2.0,
            min_solidity=0.8
    ):
        result = []

        frame = self.cam.get_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, self.bg_gray)
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            similarity = cv2.matchShapes(self.master_contour, cnt, cv2.CONTOURS_MATCH_I1, 0.0)
            if similarity > minimal_similarity:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > max_aspect_ratio:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < min_solidity:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            uv1 = np.array([[cx], [cy], [1]])
            xyz_cam = self.camera_height * (self.K_inv @ uv1)

            xyz_cam_homogeneous = np.vstack((xyz_cam, [[1]]))
            xyz_world_homogeneous = self.rt_matrix @ xyz_cam_homogeneous
            xyz_world = xyz_world_homogeneous[:3].flatten()

            result.append((cnt, xyz_world))

        return result
