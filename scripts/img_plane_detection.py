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
PLANE_SAVE_FILE = pathlib.Path("plane.json")

# Новые переменные для функционала
mouse_pos = (0, 0)
click_positions = []
plane_equation = None
reference_rotation = None
reference_tvec = None
projection_matrix = None


def mouse_callback(event, x, y, flags, param):
    global mouse_pos, click_positions
    mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        click_positions.append((x, y))


def pixel_to_world(point_px, intrinsic_matrix, distortion_vector, rvec, tvec, plane_normal, plane_point):
    # Преобразуем точку из пикселей в луч в 3D пространстве
    undistorted_point = cv2.undistortPoints(np.array([[point_px]], dtype=np.float32),
                                            intrinsic_matrix, distortion_vector)
    undistorted_point = undistorted_point[0][0]

    # Направляющий вектор луча (в системе координат камеры)
    ray_dir = np.array([undistorted_point[0], undistorted_point[1], 1.0])
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    # Точка на луче (в системе координат камеры)
    ray_origin = np.array([0, 0, 0])

    # Переводим луч в мировую систему координат
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    world_ray_dir = rotation_matrix.T @ ray_dir
    world_ray_origin = rotation_matrix.T @ (ray_origin - tvec)

    # Находим пересечение луча с плоскостью
    denom = np.dot(plane_normal, world_ray_dir)
    if abs(denom) > 1e-6:
        t = np.dot(plane_point - world_ray_origin, plane_normal) / denom
        intersection = world_ray_origin + t * world_ray_dir
        return intersection
    return None


def calculate_projection_matrix(intrinsic_matrix, distortion_vector, rvec, tvec, plane_normal, plane_point):
    # Получаем матрицу поворота
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Матрица перехода от мировых координат к координатам камеры
    extrinsic = np.hstack((rotation_matrix, tvec.reshape(3, 1)))

    # Матрица проекции (3x4)
    P = intrinsic_matrix @ extrinsic

    # Находим преобразование для плоскости
    # Уравнение плоскости: n·X + d = 0
    d = -np.dot(plane_normal, plane_point)
    H = P[:, [0, 1, 3]] - (P[:, 2] * np.array([plane_normal[0], plane_normal[1], 0]) / (-d))

    # Добавляем компоненту для z=0
    M = np.linalg.inv(H)
    return M.tolist()


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
                reference_tvec = markers[3]['tvec'][0]  # Используем [0] чтобы получить 1D массив

                # Получаем матрицу поворота для опорного маркера
                reference_rotation, _ = cv2.Rodrigues(reference_rvec)

            # Если есть опорная система координат, вычисляем относительные координаты
            if reference_rotation is not None and reference_tvec is not None:
                for marker_id, data in markers.items():
                    if marker_id == 3:
                        # Для опорного маркера координаты (0,0,0)
                        world_coords = np.array([0, 0, 0])
                    else:
                        # Вычисляем относительные координаты
                        relative_tvec = data['tvec'][0] - reference_tvec

                        # Преобразуем в мировую систему координат (поворачиваем обратно)
                        world_coords = np.dot(reference_rotation.T, relative_tvec)

                    # Сохраняем мировые координаты
                    data['world_coords'] = world_coords
                    plane_points.append(world_coords)  # Добавляем точку для плоскости

                    # Вычисляем центр маркера для отображения текста
                    center = np.mean(data['corners'][0], axis=0).astype(int)
                    corner = tuple(center)

                    coord_text = f"ID {marker_id}: ({world_coords[0]:.3f}, {world_coords[1]:.3f}, {world_coords[2]:.3f})"
                    cv2.putText(
                        frame, coord_text, (corner[0], corner[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                    )
                    # Рисуем точку в центре маркера
                    cv2.circle(frame, corner, 3, (0, 255, 0), -1)

                # Вычисляем уравнение плоскости, если есть 3 или более точек
                if len(plane_points) >= 3:
                    points = np.array(plane_points)
                    centroid = np.mean(points, axis=0)
                    centered = points - centroid
                    _, _, vh = np.linalg.svd(centered)
                    normal = vh[2, :]
                    d = -np.dot(normal, centroid)
                    plane_equation = (normal, d, centroid)  # Сохраняем нормаль, d и центроид

                    # Вычисляем матрицу проекции
                    projection_matrix = calculate_projection_matrix(
                        intrinsic_matrix, distortion_vector,
                        reference_rvec, reference_tvec,
                        normal, centroid
                    )

        # 1. Отображаем уравнение плоскости в верхнем левом углу
        if plane_equation is not None:
            normal, d, _ = plane_equation
            eq_text = f"Plane: {normal[0]:.2f}x + {normal[1]:.2f}y + {normal[2]:.2f}z + {d:.2f} = 0"
            cv2.putText(frame, eq_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Отображаем информацию о матрице проекции
            cv2.putText(frame, "Press 's' to save projection matrix", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 2. Отображаем позицию мыши
        cv2.putText(frame, f"Mouse: {mouse_pos[0]}, {mouse_pos[1]}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 3. Обрабатываем клики мыши и отображаем точки
        if reference_rotation is not None and reference_tvec is not None and plane_equation is not None:
            normal, d, centroid = plane_equation
            plane_point = np.array([0, 0, -d / normal[2]]) if normal[2] != 0 else np.array([0, 0, 0])

            for click in click_positions:
                cv2.circle(frame, click, 5, (0, 0, 255), -1)

                # Преобразуем координаты клика в мировые координаты
                if projection_matrix is not None:
                    # Используем матрицу проекции для преобразования
                    uv1 = np.array([click[0], click[1], 1])
                    world_point = projection_matrix @ uv1
                    world_point /= world_point[2]  # Нормализуем
                else:
                    # Альтернативный метод (если матрица проекции не вычислена)
                    world_point = pixel_to_world(click, intrinsic_matrix, distortion_vector,
                                                 reference_rvec, reference_tvec, normal, plane_point)

                if world_point is not None:
                    coord_text = f"({world_point[0]:.3f}, {world_point[1]:.3f}, {world_point[2]:.3f})"
                    cv2.putText(frame, coord_text, (click[0] + 10, click[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Показываем результат
        cv2.imshow("ArUco Marker Detection", cv2.resize(frame, OTHER_WINDOW_SIZE))

        key = cv2.waitKey(1) & 0xFF
        # Выход по нажатию 'q'
        if key == ord(OTHER_QUIT_KEY):
            break
        # Сохранение матрицы проекции по нажатию 's'
        elif key == ord('s') and projection_matrix is not None:
            data_to_save = {
                "projection_matrix": projection_matrix,
                "plane_normal": normal.tolist(),
                "plane_d": float(d)
            }
            with open(PLANE_SAVE_FILE, "w") as f:
                json.dump(data_to_save, f)
            print(f"Projection matrix saved to {PLANE_SAVE_FILE}")

    cv2.destroyAllWindows()