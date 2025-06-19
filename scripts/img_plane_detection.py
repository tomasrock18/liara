import json
import cv2
import numpy as np
import pathlib
import liara

# Параметры
CAMERA_ID = 1
CAMERA_FRAME_SIZE = (1080, 1920)
CAMERA_INITIAL_FOCUS = 40
TARGET_IDS = {3, 6, 24, 27}  # ID маркеров, которые мы ищем
BOARD_MARKER_LENGTH_M = 0.025
OTHER_QUIT_KEY = "q"
OTHER_WINDOW_SIZE = (1920, 1080)
OTHER_CALIBRATION_RESULTS_FILE_PATH = pathlib.Path("calibration_results.json")

# Новые переменные для функционала
mouse_pos = (0, 0)
click_positions = []
plane_equation = None


def mouse_callback(event, x, y, flags, param):
    global mouse_pos, click_positions
    mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        click_positions.append((x, y))


if __name__ == "__main__":
    # Загрузка параметров камеры
    with open(OTHER_CALIBRATION_RESULTS_FILE_PATH, "r") as file:
        calibration_results = json.load(file)
    intrinsic_matrix = np.array(calibration_results["intrinsic_matrix"])
    distortion_vector = np.array(calibration_results["distortion_vector"])

    # Инициализация детектора
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    # Инициализация камеры
    cam = liara.CameraLab(CAMERA_ID, CAMERA_FRAME_SIZE)
    cam.focus = CAMERA_INITIAL_FOCUS

    # Переменные для хранения опорной системы координат (маркер 3)
    reference_rvec = None
    reference_tvec = None
    reference_rotation = None

    # Создаем окно и устанавливаем обработчик мыши
    cv2.namedWindow("ArUco Marker Detection")
    cv2.setMouseCallback("ArUco Marker Detection", mouse_callback)

    while True:
        frame = cam.get_frame()
        corners, ids, _ = detector.detectMarkers(frame)

        # Новые переменные для хранения точек плоскости
        plane_points = []

        if ids is not None:
            # Словарь для хранения позиций маркеров
            markers = {}

            # Оценка позы для всех маркеров
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, BOARD_MARKER_LENGTH_M, intrinsic_matrix, distortion_vector)

            # Сохраняем данные только для целевых маркеров
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in TARGET_IDS:
                    markers[marker_id] = {
                        'corners': corners[i],
                        'rvec': rvecs[i],
                        'tvec': tvecs[i],
                        'center': tvecs[i][0]  # Центр маркера в координатах камеры
                    }

            # Если найден маркер 3, используем его как опорный
            if 3 in markers:
                reference_rvec = markers[3]['rvec']
                reference_tvec = markers[3]['tvec']

                # Получаем матрицу поворота для опорного маркера
                reference_rotation, _ = cv2.Rodrigues(reference_rvec)

            # Если есть опорная система координат, вычисляем относительные координаты
            if reference_rotation is not None:
                for marker_id, data in markers.items():
                    if marker_id == 3:
                        # Для опорного маркера координаты (0,0,0)
                        world_coords = np.array([0, 0, 0])
                    else:
                        # Вычисляем относительные координаты
                        relative_tvec = data['tvec'][0] - reference_tvec[0]

                        # Преобразуем в мировую систему координат (поворачиваем обратно)
                        world_coords = np.dot(reference_rotation.T, relative_tvec)

                    # Сохраняем мировые координаты
                    data['world_coords'] = world_coords
                    plane_points.append(world_coords)  # Добавляем точку для плоскости

                    # Вычисляем центр маркера для отображения текста
                    center = np.mean(data['corners'][0], axis=0).astype(int)
                    corner = tuple(center)

                    coord_text = f"ID {marker_id}: ({world_coords[1]:.3f}, {world_coords[0]:.3f}, {world_coords[2]:.3f})"
                    cv2.putText(
                        frame, coord_text, (corner[0], corner[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                    )
                    # Рисуем точку в центре маркера
                    cv2.circle(frame, corner, 3, (0, 255, 0), -1)

                # Вычисляем уравнение плоскости, если есть 4 точки
                if len(plane_points) == 4:
                    points = np.array(plane_points)
                    centroid = np.mean(points, axis=0)
                    centered = points - centroid
                    _, _, vh = np.linalg.svd(centered)
                    normal = vh[2, :]
                    d = -np.dot(normal, centroid)
                    plane_equation = np.append(normal, d)

        # 1. Отображаем уравнение плоскости в верхнем левом углу
        if plane_equation is not None:
            eq_text = f"Plane: {plane_equation[0]:.2f}x + {plane_equation[1]:.2f}y + {plane_equation[2]:.2f}z + {plane_equation[3]:.2f} = 0"
            cv2.putText(frame, eq_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 2. Отображаем позицию мыши
        cv2.putText(frame, f"Mouse: {mouse_pos[0]}, {mouse_pos[1]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)

        # 3. Обрабатываем клики мыши и отображаем точки
        for click in click_positions:
            cv2.circle(frame, click, 5, (0, 0, 255), -1)

            if plane_equation is not None:
                # Преобразуем координаты клика в мировые координаты
                # (Это упрощенное преобразование, может потребоваться доработка)
                z = 0  # Предполагаем, что точка лежит на плоскости (z=0)
                x = (click[0] - frame.shape[1] / 2) * 0.001  # Примерное преобразование
                y = (click[1] - frame.shape[0] / 2) * 0.001  # Примерное преобразование

                # Используем уравнение плоскости для уточнения координат
                if plane_equation[2] != 0:
                    z = (-plane_equation[3] - plane_equation[0] * x - plane_equation[1] * y) / plane_equation[2]

                coord_text = f"({x:.3f}, {y:.3f}, {z:.3f})"
                cv2.putText(frame, coord_text, (click[0] + 10, click[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Показываем результат
        cv2.imshow("ArUco Marker Detection", cv2.resize(frame, OTHER_WINDOW_SIZE))

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord(OTHER_QUIT_KEY):
            break

    cv2.destroyAllWindows()