import cv2
import numpy

# 1) Объявляем входные данные скрипта
#####################################################################
CAMERA_ID = 1
CAMERA_FRAME_SIZE = (1920, 1080)
BOARD_SIZE = (7, 9)
BOARD_MARKER_LENGTH_M = 0.025
BOARD_SQUARE_LENGTH_M = 0.03
BOARD_ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
PLANE_MARKERS_IDS = (3, 6, 24, 27)
PLANE_ANCHOR_MARKER_ID = 3
#####################################################################


# 2) Выполняем настройку камеры
#####################################################################
# Предварительная настройка
cam = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_SIZE[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_SIZE[1])
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

# Настройка окна
camera_setting_window_name = "Camera set up"
focus_option_name = "Focus"
cv2.namedWindow(camera_setting_window_name, cv2.WINDOW_NORMAL)


def on_focus_change(value) -> None:
    cam.set(cv2.CAP_PROP_FOCUS, int(value))


cv2.createTrackbar(
    focus_option_name,
    camera_setting_window_name,
    int(cam.get(cv2.CAP_PROP_FOCUS)),
    255,
    on_focus_change
)

# Запуск цикла работы камеры
while True:
    ret, frame = cam.read()
    if not ret:
        raise Exception("Камера не вернула картинку")
    cv2.putText(
        frame,
        f"Current focus is: {int(cam.get(cv2.CAP_PROP_FOCUS))}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        1
    )
    cv2.putText(
        frame,
        f"Press 'Enter' to continue",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
    cv2.imshow(camera_setting_window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord("\r"):
        cv2.destroyAllWindows()
        break
#####################################################################

# 3) Выполняем калибровку камеры
#####################################################################
# Создание объекта доски ChAruco
calibration_board = cv2.aruco.CharucoBoard(
    size=BOARD_SIZE,
    markerLength=BOARD_MARKER_LENGTH_M,
    squareLength=BOARD_SQUARE_LENGTH_M,
    dictionary=BOARD_ARUCO_DICTIONARY,
)

# Инициализация массивов данных для калибровки
all_charuco_corners = []
all_charuco_ids = []

# Инициализация детектора параметров калибровки
detector = cv2.aruco.CharucoDetector(calibration_board)

# Инициализация массива изображений для калибровки
calibration_images = []

# Настройка окна
window_name = "Initial Calibration"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Запуск цикла работы камеры
while True:
    ret, frame = cam.read()
    if not ret:
        raise Exception("Камера не вернула картинку")

    # Переводим в серое
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Ищем углы
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(grey)

    # Отрисовка ну и обработка + разметка
    if charuco_ids is not None and charuco_corners is not None:
        cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

    cv2.putText(
        frame,
        f"Frames for calibration: {len(all_charuco_corners)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        1
    )
    cv2.putText(
        frame,
        "Press 'a' to add frame for calibration",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
    cv2.putText(
        frame,
        "Press 'Enter' to start calibration",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("a"):
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
    elif key == ord("\r"):
        projection_error_calibration, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners,
            all_charuco_ids,
            calibration_board,
            grey.shape,
            cameraMatrix=None,
            distCoeffs=None,
        )
        cv2.destroyAllWindows()
        break

#####################################################################

# 4) Выполняем построение плоскости преобразования
#####################################################################
# Инициализация детектора ArUco
aruco_detector = cv2.aruco.ArucoDetector(BOARD_ARUCO_DICTIONARY, cv2.aruco.DetectorParameters())

# Инициализация окна
window_name = "Plane Definition"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cam.read()
    if not ret:
        raise Exception("Камера не вернула картинку")

    # Обнаружение маркеров
    corners, ids, _ = aruco_detector.detectMarkers(frame)

    # Определяем вектора поворота и смещений каждого маркера относительно камеры
    m_rvecs, m_tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, BOARD_MARKER_LENGTH_M, camera_matrix, dist_coeffs
    )

    # Находим вектора опорного маркера
    if m_rvecs is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == PLANE_ANCHOR_MARKER_ID:
                anchor_rvec = m_rvecs[i]
                anchor_tvec = m_tvecs[i]
                break

    # Если нашли опорный маркер, вычисляем плоскость
    plane_points = []
    if anchor_rvec is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in PLANE_MARKERS_IDS:
                # Преобразуем координаты маркеров, относительно опорного
                anchor_rot_matrix, _ = cv2.Rodrigues(anchor_rvec)
                marker_tvec = m_tvecs[i][0] - anchor_tvec[0]
                world_coords = anchor_rot_matrix.T @ marker_tvec
                plane_points.append(world_coords)

    # Вычисляем уравнение плоскости
    if len(plane_points) > 0:
        points = numpy.array(plane_points)
        centroid = numpy.mean(points, axis=0)
        centered = points - centroid
        _, _, vh = numpy.linalg.svd(centered)
        normal = vh[2, :]
        d = -numpy.dot(normal, centroid)

        # Отображаем уравнение плоскости
        cv2.putText(frame,
                    f"Plane: {normal[0]:.2f}x + {normal[1]:.2f}y + {normal[2]:.2f}z + {d:.2f} = 0",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(
        frame,
        f"Press 'Enter' to continue",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord("\r"):
        cv2.destroyAllWindows()

#####################################################################
