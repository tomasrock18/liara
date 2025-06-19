import cv2
import numpy as np


def is_good_for_charuco_calibration(img: np.ndarray, charuco_board: cv2.aruco.CharucoBoard) -> bool:
    """
    Функция проверки изображения на пригодность для калибровки по методу ChAruco.

    :param img: Изображение для проверки.
    :param charuco_board: Объект доски, которая распознаётся в процессе калибровки.
    :return: Флаг, если True, тогда изображение пригодно для использования в процессе калибровке.
    """
    # Инициализация детектора параметров калибровки
    detector = cv2.aruco.CharucoDetector(charuco_board)

    # Сбор данных из изображения
    aruco_corners, aruco_ids, _, _ = detector.detectBoard(img)

    # Если не найдены все параметры, то изображение считается непригодным
    if aruco_corners is not None and len(aruco_corners) > 0:
        return True
    else:
        return False
