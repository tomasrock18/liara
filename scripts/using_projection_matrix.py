import json
import cv2
import numpy as np
import pathlib
import liara

# Параметры камеры (как в первом скрипте)
CAMERA_ID = 1
CAMERA_FRAME_SIZE = (1080, 1920)
CAMERA_INITIAL_FOCUS = 40
OTHER_WINDOW_SIZE = (1920, 1080)
PLANE_FILE = pathlib.Path("plane.json")
QUIT_KEY = "q"

# Глобальные переменные
mouse_pos = (0, 0)
click_positions = []
projection_matrix = None
plane_equation = None  # Для хранения уравнения плоскости


def mouse_callback(event, x, y, flags, param):
    global mouse_pos, click_positions
    mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        click_positions.append((x, y))


def load_projection_data():
    """Загружает матрицу проекции и уравнение плоскости"""
    with open(PLANE_FILE, "r") as f:
        data = json.load(f)
    return (
        np.array(data["projection_matrix"]),
        np.array(data["plane_normal"]),
        data["plane_d"]
    )


def pixel_to_world_corrected(u, v, M, normal, d):
    """Корректное преобразование пиксельных координат в мировые с учетом плоскости"""
    # Шаг 1: Преобразуем (u,v) в луч в 3D пространстве
    uv1 = np.array([u, v, 1], dtype=np.float32)
    xyz = M @ uv1

    # Шаг 2: Находим точку пересечения с плоскостью
    # Плоскость: normal·X + d = 0
    t = - (np.dot(normal, xyz[:3]) + d) / np.dot(normal, xyz[3:6])
    world_point = xyz[:3] + t * xyz[3:6]

    return world_point


if __name__ == "__main__":
    # Загружаем данные проекции
    try:
        projection_matrix, plane_normal, plane_d = load_projection_data()
        print("Данные проекции успешно загружены")
        print(f"Матрица проекции:\n{projection_matrix}")
        print(
            f"Уравнение плоскости: {plane_normal[0]:.2f}x + {plane_normal[1]:.2f}y + {plane_normal[2]:.2f}z + {plane_d:.2f} = 0")
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        exit()

    # Инициализация камеры
    cam = liara.CameraLab(CAMERA_ID, CAMERA_FRAME_SIZE)
    cam.focus = CAMERA_INITIAL_FOCUS

    # Создаем окно
    cv2.namedWindow("Camera Projection Viewer")
    cv2.setMouseCallback("Camera Projection Viewer", mouse_callback)

    while True:
        frame = cam.get_frame()

        # Отображаем информацию
        cv2.putText(frame, f"Mouse: {mouse_pos[0]}, {mouse_pos[1]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "LMB: add point, 'c': clear, 'q': quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Обрабатываем клики
        for click in click_positions:
            u, v = click
            world_point = pixel_to_world_corrected(u, v, projection_matrix, plane_normal, plane_d)

            # Рисуем точку и подпись
            cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
            coord_text = f"({world_point[0]:.3f}, {world_point[1]:.3f}, {world_point[2]:.3f})"
            cv2.putText(frame, coord_text, (u + 10, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Camera Projection Viewer", cv2.resize(frame, OTHER_WINDOW_SIZE))

        key = cv2.waitKey(1) & 0xFF
        if key == ord(QUIT_KEY):
            break
        elif key == ord('c'):
            click_positions = []

    cam.release()
    cv2.destroyAllWindows()