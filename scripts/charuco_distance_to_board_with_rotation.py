import json

import cv2
import numpy
import pathlib

import liara

# Инициализация параметров камеры и детектируемой доски
CAMERA_ID = 1
CAMERA_FRAME_SIZE = (1080, 1920)
CAMERA_INITIAL_FOCUS = 40
BOARD_SIZE = (7, 9)
BOARD_MARKER_LENGTH_M = 0.025
BOARD_SQUARE_LENGTH_M = 0.03
BOARD_ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

# Инициализация прочих переменных
OTHER_QUIT_KEY = "q"
OTHER_WINDOW_SIZE = (1000, 500)
OTHER_CALIBRATION_RESULTS_FILE_PATH = pathlib.Path("calibration_results.json")

if __name__ == "__main__":
    # Чтение данных калибровки камеры
    with open(OTHER_CALIBRATION_RESULTS_FILE_PATH, "r") as file:
        calibration_results = json.load(file)
    intrinsic_matrix = numpy.array(calibration_results["intrinsic_matrix"])
    distortion_vector = numpy.array(calibration_results["distortion_vector"])

    # Инициализация объекта калибровочной доски
    board = cv2.aruco.CharucoBoard(
        size=BOARD_SIZE,
        squareLength=BOARD_SQUARE_LENGTH_M,
        markerLength=BOARD_MARKER_LENGTH_M,
        dictionary=BOARD_ARUCO_DICTIONARY
    )

    # Инициализация детектора
    detector = cv2.aruco.CharucoDetector(board)

    # Инициализация камеры
    cam = liara.CameraLab(CAMERA_ID, CAMERA_FRAME_SIZE)
    cam.focus = CAMERA_INITIAL_FOCUS

    # Процесс обработки
    while True:
        # Захват кадра
        frame = cam.get_frame()

        # Обнаружение доски
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )

        # Если доска обнаружена, выполнение оценки позиции, с предварительной инициализацией параметров
        is_estimated = False
        rotation_vector = None
        transmission_vector = None
        if charuco_corners is not None and len(charuco_corners) > 0:
            is_estimated, rotation_vector, transmission_vector = cv2.aruco.estimatePoseCharucoBoard(
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids,
                board=board,
                cameraMatrix=intrinsic_matrix,
                distCoeffs=distortion_vector,
                rvec=None,
                tvec=None
            )

        # Проверка успешности выполнения оценки позиции
        if is_estimated:
            # Расчёт расстояния до доски
            distance_to_board = numpy.linalg.norm(transmission_vector)

            # Расчёт углов поворота камеры
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            # Расчёт масштабного коэффициента
            s = numpy.sqrt(
                rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0]
            )

            # Проверка на сингулярность
            is_singular = s < 1e-6

            # Расчёт углов
            if not is_singular:
                x = numpy.rad2deg(numpy.arcsin(-rotation_matrix[2, 0]))
                y = numpy.rad2deg(numpy.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
                z = numpy.rad2deg(numpy.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
            else:
                x = numpy.rad2deg(numpy.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
                y = numpy.rad2deg(numpy.arctan2(-rotation_matrix[2, 0], s))
                z = numpy.rad2deg(0)

            # Разметка кадра
            cv2.putText(
                frame,
                f"Distance: {distance_to_board:.3f} m",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            for rotation_axis, rotation_angle in {"x0": x, "y1": y, "z2": z}.items():
                cv2.putText(
                    frame,
                    f"Rotation {rotation_axis[0]}: {rotation_angle:.3f} deg",
                    (20, 80 + (40 * int(rotation_axis[1]))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

            # Визуализация осей доски
            cv2.drawFrameAxes(
                frame,
                intrinsic_matrix,
                distortion_vector,
                rotation_vector,
                transmission_vector,
                0.1
            )

            # Отрисовка обнаруженных углов доски
            if charuco_corners is not None and len(charuco_corners) > 0:
                cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

            # Обновление кадра
            cv2.imshow("Charuco Board Detection", cv2.resize(frame, OTHER_WINDOW_SIZE))
            if cv2.waitKey(1) & 0xFF == ord(OTHER_QUIT_KEY):
                break

    cv2.destroyAllWindows()
