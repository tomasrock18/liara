import abc

import cv2
import numpy


class CalibrationBase(abc.ABC):
    @abc.abstractmethod
    def get_calibration_matrix(self) -> list[list[float]]:
        ...


class CalibrationChAruco(CalibrationBase):
    def __init__(self, images: list[numpy.ndarray], board_template: cv2.aruco.CharucoBoard):
        corners_all = []
        markers_ids_all = []
        interpolated_corners_all = []
        interpolated_ids_all = []

        detector = cv2.aruco.CharucoDetector(board_template)

        for num, img in enumerate(images):

            interpolated_corners, interpolated_ids, corners, marker_ids = detector.detectBoard(
                img
            )

            for parameter in (interpolated_corners, interpolated_ids, corners, marker_ids):
                if parameter is None:
                    raise ValueError(f"Bad image {num}")

            corners_all.append(corners)
            markers_ids_all.append(marker_ids)
            interpolated_corners_all.append(interpolated_corners)
            interpolated_ids_all.append(interpolated_ids)

        calibration_results = cv2.aruco.calibrateCameraCharuco(
            interpolated_corners_all,
            interpolated_ids_all,
            board=board_template,
            imageSize=images[0].shape,
            cameraMatrix=None,
            distCoeffs=None,
        )

        self._calibration_matrix: list[list[float]] = calibration_results[1].tolist()

    def get_calibration_matrix(self) -> list[list[float]]:
        return self._calibration_matrix
