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

# Глобальные переменные для плоскости
plane_normal = None
plane_d = None
anchor_rot_matrix = None
anchor_tvec = None

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
        plane_normal = vh[2, :]
        plane_d = -numpy.dot(plane_normal, centroid)

        # Отображаем уравнение плоскости
        cv2.putText(frame,
                    f"Plane: {plane_normal[0]:.2f}x + {plane_normal[1]:.2f}y + {plane_normal[2]:.2f}z + {plane_d:.2f} = 0",
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

# 7) Детекция объектов через вычитание фона с ROI
#####################################################################
window_name = "Object Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Инициализация параметров
threshold_value = 25
min_area_value = 500
roi_top = 0
roi_bottom = 100
roi_left = 0
roi_right = 100

# Глобальные переменные для эталонного контура
template_contour = None
similarity_threshold = 0.7


# Создание трекбаров
def on_threshold_change(val):
    global threshold_value
    threshold_value = val


def on_min_area_change(val):
    global min_area_value
    min_area_value = val


def on_roi_top_change(val):
    global roi_top
    roi_top = val


def on_roi_bottom_change(val):
    global roi_bottom
    roi_bottom = val


def on_roi_left_change(val):
    global roi_left
    roi_left = val


def on_roi_right_change(val):
    global roi_right
    roi_right = val


cv2.createTrackbar('Threshold', window_name, threshold_value, 255, on_threshold_change)
cv2.createTrackbar('Min Area', window_name, min_area_value, 5000, on_min_area_change)
cv2.createTrackbar('ROI Top', window_name, roi_top, 100, on_roi_top_change)
cv2.createTrackbar('ROI Bottom', window_name, roi_bottom, 100, on_roi_bottom_change)
cv2.createTrackbar('ROI Left', window_name, roi_left, 100, on_roi_left_change)
cv2.createTrackbar('ROI Right', window_name, roi_right, 100, on_roi_right_change)

while True:
    ret, frame = cam.read()
    if not ret:
        raise Exception("Камера не вернула картинку")

    # Преобразование значений ROI в относительные координаты
    height, width = frame.shape[:2]
    top = int(height * roi_top / 100)
    bottom = int(height * roi_bottom / 100)
    left = int(width * roi_left / 100)
    right = int(width * roi_right / 100)

    # Ограничение значений ROI
    top = max(0, min(top, height - 1))
    bottom = max(top + 1, min(bottom, height))
    left = max(0, min(left, width - 1))
    right = max(left + 1, min(right, width))

    # Выделение ROI
    roi_frame = frame[top:bottom, left:right]
    roi_bg = bg[top:bottom, left:right]

    # 1. Преобразуем ROI в grayscale
    gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    gray_bg = cv2.cvtColor(roi_bg, cv2.COLOR_BGR2GRAY)

    # 2. Вычисляем абсолютную разницу
    diff = cv2.absdiff(gray_bg, gray_frame)

    # 3. Бинаризация разницы
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    # 4. Морфологические операции
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 5. Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Отрисовка результатов
    result_frame = frame.copy()

    # Отрисовка линий ROI
    cv2.line(result_frame, (left, top), (right, top), (255, 255, 0), 2)
    cv2.line(result_frame, (right, top), (right, bottom), (255, 255, 0), 2)
    cv2.line(result_frame, (right, bottom), (left, bottom), (255, 255, 0), 2)
    cv2.line(result_frame, (left, bottom), (left, top), (255, 255, 0), 2)

    for contour in contours:
        if cv2.contourArea(contour) < min_area_value:
            continue

        # Корректировка координат контура относительно полного кадра
        offset_contour = contour + (left, top)

        # Отрисовка контура
        cv2.drawContours(result_frame, [offset_contour], -1, (0, 255, 0), 2)

        # Центр масс
        M = cv2.moments(offset_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(result_frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(result_frame, f"({cX}, {cY})", (cX - 50, cY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Сохраняем первый подходящий контур как эталонный
            if template_contour is None:
                template_contour = offset_contour
                cv2.putText(result_frame, "Template saved!", (cX, cY + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Отображение параметров
    info_y = 30
    cv2.putText(result_frame, f"Threshold: {threshold_value}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(result_frame, f"Min Area: {min_area_value}", (10, info_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(result_frame, f"ROI: {left}:{right}, {top}:{bottom}", (10, info_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(result_frame, "Press 'Enter' to continue", (10, info_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow(window_name, result_frame)
    if cv2.waitKey(1) & 0xFF == ord("\r"):
        cv2.destroyAllWindows()
        break
#####################################################################

# 8) Поиск похожих контуров и вычисление их 3D координат на плоскости
#####################################################################
window_name = "3D Coordinates Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Параметры для сравнения контуров
MIN_SIMILARITY = 0.1  # Минимальный коэффициент схожести контуров


def contour_similarity(contour1, contour2):
    """Вычисляет коэффициент схожести двух контуров"""
    return cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I2, 0)


def pixel_to_3d(pixel_x, pixel_y, normal, d, camera_matrix):
    """Преобразует координаты пикселя в 3D координаты на плоскости"""
    # Преобразование пикселя в луч в 3D (в системе координат камеры)
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # Направление луча (нормированное)
    x = (pixel_x - cx) / fx
    y = (pixel_y - cy) / fy
    z = 1.0
    ray_dir = numpy.array([x, y, z])
    ray_dir = ray_dir / numpy.linalg.norm(ray_dir)  # Нормируем

    # Находим пересечение луча с плоскостью
    C = numpy.array([0, 0, 0])  # Позиция камеры
    denominator = numpy.dot(normal, ray_dir)

    if abs(denominator) < 1e-6:  # Луч параллелен плоскости
        return None

    lambda_val = -(numpy.dot(normal, C) + d) / denominator
    point_3d = C + lambda_val * ray_dir

    return point_3d


while True:
    ret, frame = cam.read()
    if not ret:
        raise Exception("Камера не вернула картинку")

    # 1. Преобразуем кадр в grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Вычисляем абсолютную разницу с фоном
    gray_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_bg, gray_frame)

    # 3. Бинаризация разницы
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    # 4. Морфологические операции
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 5. Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Отрисовка результатов
    result_frame = frame.copy()

    if template_contour is not None and plane_normal is not None and plane_d is not None:
        for contour in contours:
            if cv2.contourArea(contour) < min_area_value:
                continue

            # Сравниваем контур с эталонным
            similarity = contour_similarity(template_contour, contour)
            if similarity > MIN_SIMILARITY:
                continue  # Пропускаем недостаточно похожие контуры

            # Находим центр масс
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Вычисляем 3D координаты на плоскости
                point_3d = pixel_to_3d(cX, cY, plane_normal, plane_d, camera_matrix)
                if point_3d is not None:
                    # Преобразуем координаты относительно опорного маркера
                    if anchor_rot_matrix is not None and anchor_tvec is not None:
                        world_coords = anchor_rot_matrix.T @ (point_3d - anchor_tvec[0])
                    else:
                        world_coords = point_3d

                    # Отрисовка контура
                    cv2.drawContours(result_frame, [contour], -1, (0, 255, 0), 2)

                    # Отрисовка центра масс и координат
                    cv2.circle(result_frame, (cX, cY), 5, (0, 0, 255), -1)
                    cv2.putText(result_frame,
                                f"X:{world_coords[0]:.1f} Y:{world_coords[1]:.1f} Z:{world_coords[2]:.1f}",
                                (cX - 100, cY - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Отображение информации
    cv2.putText(result_frame, "Detecting similar objects on plane", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(result_frame, "Press 'Enter' to exit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow(window_name, result_frame)
    if cv2.waitKey(1) & 0xFF == ord("\r"):
        cv2.destroyAllWindows()
        break

cam.release()
#####################################################################