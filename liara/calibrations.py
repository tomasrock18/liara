import abc

import cv2
import numpy

from .utils import is_good_for_charuco_calibration


class CalibrationBase(abc.ABC):
    """
    Интерфейс класса калибровки.

    """

    @property
    @abc.abstractmethod
    def intrinsic_parameters(self) -> numpy.ndarray:
        """
        Атрибут матрицы внутренних параметров камеры.

        :return: Матрица внутренних параметров камеры.
        """

    @property
    @abc.abstractmethod
    def distortion_coefficients(self) -> numpy.ndarray:
        """
        Атрибут вектора искажений камеры.

        :return: Вектор-столбец искажений камеры.
        """


class CalibrationChAruco(CalibrationBase):
    """
    Класс калибровки камеры по методу ChAruco.

    """

    def __init__(
            self,
            images: list[numpy.ndarray],
            board: cv2.aruco.CharucoBoard
    ):
        # Инициализация массивов данных для калибровки
        all_aruco_corners = []
        all_aruco_ids = []

        # Инициализация детектора параметров калибровки
        detector = cv2.aruco.CharucoDetector(board)

        # Подготовка данных для калибровки
        for image in images:
            # Сбор данных из изображения
            aruco_corners, aruco_ids, _, _ = detector.detectBoard(image)

            # Если найдены все параметры, данные изображения сохраняются
            if is_good_for_charuco_calibration(image, board):
                all_aruco_corners.append(aruco_corners)
                all_aruco_ids.append(aruco_ids)

        # Если не найдено ни одного параметра, выбрасывается исключения
        assert len(all_aruco_corners) == len(all_aruco_ids)
        assert len(all_aruco_corners) != 0

        # Калибровка с последующей инициализацией атрибутов
        _, self._intrinsic_parameters, self._distortion_coefficients, _, _ = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_aruco_corners,
            charucoIds=all_aruco_ids,
            board=board,
            imageSize=images[0].shape,
            cameraMatrix=None,
            distCoeffs=None,
        )

    def intrinsic_parameters(self) -> numpy.ndarray:
        return self._intrinsic_parameters

    def distortion_coefficients(self) -> numpy.ndarray:
        return self._distortion_coefficients
