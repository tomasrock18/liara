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
    anchor_rvec = None
    anchor_tvec = None
    if m_rvecs is not None and ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == PLANE_ANCHOR_MARKER_ID:
                anchor_rvec = m_rvecs[i]
                anchor_tvec = m_tvecs[i]
                # Отрисовка осей для якорного маркера
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, anchor_rvec, anchor_tvec,
                                  BOARD_MARKER_LENGTH_M * 1.5)
                break

    # Если нашли опорный маркер, вычисляем плоскость и отрисовываем мировые координаты
    plane_points = []
    if anchor_rvec is not None and m_rvecs is not None and ids is not None:
        anchor_rot_matrix, _ = cv2.Rodrigues(anchor_rvec)
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in PLANE_MARKERS_IDS:
                # Преобразуем координаты маркеров, относительно опорного
                marker_tvec = m_tvecs[i][0] - anchor_tvec[0]
                world_coords = anchor_rot_matrix.T @ marker_tvec
                plane_points.append(world_coords)

                # Отрисовка мировых координат только для целевых маркеров
                text_pos = tuple(map(int, corners[i][0][0]))
                cv2.putText(frame, f"ID:{marker_id} X:{world_coords[0]:.3f}", (text_pos[0], text_pos[1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Y:{world_coords[1]:.3f}", (text_pos[0], text_pos[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Z:{world_coords[2]:.3f}", (text_pos[0], text_pos[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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
        break

#####################################################################

# 6) Выполняем извлечение заднего фона для обнаружения опорного контура
#####################################################################
while True:
    ret, frame = cam.read()
    if not ret:
        raise Exception("Камера не вернула картинку")

    bg = frame

    cv2.putText(
        frame,
        f"Press 'Enter' to continue",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Background Extractor", frame)
    if cv2.waitKey(1) & 0xFF == ord("\r"):
        cv2.destroyAllWindows()
        break

#####################################################################

# 7) Детекция объектов с улучшенным вычитанием фона
#####################################################################
window_name = "Object Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Инициализация детектора фона с оптимальными параметрами
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,  # Количество кадров для обучения фона
    varThreshold=16,  # Порог дисперсии для разделения фона/переднего плана
    detectShadows=False  # Не учитывать тени
)

# Ядро для морфологических операций
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Минимальная площадь контура для фильтрации шума
MIN_CONTOUR_AREA = 500

# Счетчик кадров для обучения фона
learning_frames = 0

while True:
    ret, frame = cam.read()
    if not ret:
        raise Exception("Камера не вернула картинку")

    # 1. Обучение фона (первые 50 кадров)
    if learning_frames < 50:
        bg_subtractor.apply(frame, learningRate=0.5)  # Быстрое обучение
        learning_frames += 1
        cv2.putText(frame, "Learning background...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(window_name, frame)
        cv2.waitKey(30)
        continue

    # 2. Вычитание фона
    fg_mask = bg_subtractor.apply(frame, learningRate=0.001)  # Медленная адаптация

    # 3. Улучшение маски
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)  # Удаление шума
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Заполнение дыр

    # 4. Поиск контуров
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Отрисовка результатов
    result_frame = frame.copy()

    for contour in contours:
        # Фильтрация по площади
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue

        # Отрисовка контура
        cv2.drawContours(result_frame, [contour], -1, (0, 255, 0), 2)

        # Ограничивающий прямоугольник
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Центр масс
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(result_frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(result_frame, f"({cX}, {cY})", (cX - 50, cY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(result_frame, "Press 'Enter' to exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow(window_name, result_frame)
    if cv2.waitKey(1) & 0xFF == ord("\r"):
        cv2.destroyAllWindows()
        break
#####################################################################
