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

    # Инициализация детектора
    detector = cv2.aruco.ArucoDetector(BOARD_ARUCO_DICTIONARY, cv2.aruco.DetectorParameters())

    # Инициализация камеры
    cam = liara.CameraLab(CAMERA_ID, CAMERA_FRAME_SIZE)
    cam.focus = CAMERA_INITIAL_FOCUS

    while True:
        frame = cam.get_frame()

        # Обнаружение маркеров
        corners, ids, _ = detector.detectMarkers(frame)

        # Обработка только целевых маркеров
        if ids is not None:
            for marker_id, marker_corners in zip(ids, corners):
                if marker_id in (3, 6, 24, 27):
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        marker_corners,
                        BOARD_MARKER_LENGTH_M,
                        intrinsic_matrix,
                        distortion_vector,
                    )

                    # Получаем координаты центра маркера
                    center = tvec[0][0]

                    # Отображаем оси маркера
                    cv2.drawFrameAxes(
                        frame,
                        intrinsic_matrix,
                        distortion_vector,
                        rvec,
                        tvec,
                        BOARD_MARKER_LENGTH_M * 0.5
                    )

                    # Подготавливаем текст с координатами
                    coord_text = f"ID {marker_id[0]}: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"

                    # Вычисляем позицию для текста
                    text_position = tuple(marker_corners[0][0].astype(int))

                    # Рисуем текст с координатами
                    cv2.putText(
                        frame,
                        coord_text,
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            # Отображаем все обнаруженные маркеры
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Показываем результат
        cv2.imshow("ArUco Marker Detection", cv2.resize(frame, OTHER_WINDOW_SIZE))

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord(OTHER_QUIT_KEY):
            break

    cv2.destroyAllWindows()