from .board import Board

import typing

import cv2
import numpy


class Calibration:
    MINIMAL_AMOUNT_OF_IMAGES: int = 20

    def __init__(self):
        self._template_board: Board | None = None
        self._images: list[numpy.ndarray] = []
        self._corners: list[typing.Sequence[numpy.ndarray]] = []
        self._markers_ids: list[list[int]] = []
        self._interpolated_corners: list[numpy.ndarray] = []
        self._interpolated_ids: list[list[int]] = []
        self._calibration_matrix: list[list[float]] | None = None
        self._distortion_coefficients: list[float] | None = None

    def calibrate(self) -> None:
        if self._template_board is None:
            raise ValueError("Calibration template board is None")
        if not self.is_calibration_possible():
            raise ValueError("Calibration is not possible, check images amount and repeat")

        calibration_results = cv2.aruco.calibrateCameraCharuco(
            self._interpolated_corners,
            self._interpolated_ids,
            board=self._template_board.get_board_template(),
            imageSize=self._images[0].shape[1::-1],
            cameraMatrix=None,
            distCoeffs=None,
        )
        self._calibration_matrix = calibration_results[1].tolist()
        self._distortion_coefficients = calibration_results[2].tolist()

    def is_calibration_possible(self) -> bool:
        is_enough_images: bool = len(self._images) > self.MINIMAL_AMOUNT_OF_IMAGES
        return is_enough_images

    def add_image(self, image: numpy.ndarray) -> None:
        charuco_detector = cv2.aruco.CharucoDetector(self._template_board.get_board_template())
        interpolated_corners, interpolated_ids, corners, marker_ids = charuco_detector.detectBoard(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )

        for parameter in (interpolated_corners, interpolated_ids, corners, marker_ids):
            if parameter is None:
                raise ValueError("Bad image")

        self._images.append(image)
        self._corners.append(corners)
        self._markers_ids.append(marker_ids)
        self._interpolated_corners.append(interpolated_corners)
        self._interpolated_ids.append(interpolated_ids)

    def get_image_with_markers(self, img_id: int) -> numpy.ndarray:
        result = self.images[img_id].copy()
        cv2.aruco.drawDetectedMarkers(result, self._corners[img_id], self._markers_ids[img_id])
        return result

    def get_image_with_corners(self, img_id: int) -> numpy.ndarray:
        result = self.images[img_id].copy()
        cv2.aruco.drawDetectedCornersCharuco(result, self._interpolated_corners[img_id], self._interpolated_ids[img_id])
        return result

    def get_image(self, img_id: int) -> numpy.ndarray:
        return self._images[img_id]

    def clear_images(self) -> None:
        self._images = []
        self._corners = []
        self._markers_ids = []
        self._interpolated_corners = []
        self._interpolated_ids = []

    @property
    def images(self) -> list[numpy.ndarray]:
        return self._images

    @property
    def template_board(self) -> Board | None:
        return self._template_board

    @template_board.setter
    def template_board(self, template_board: Board) -> None:
        self._template_board = template_board

    @property
    def calibration_matrix(self) -> list[list[float]]:
        return self._calibration_matrix

    @property
    def distortion_coefficients(self) -> list[float]:
        return self._distortion_coefficients
