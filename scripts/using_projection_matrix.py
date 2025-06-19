import json
import cv2
import numpy as np
import pathlib
import liara

# Параметры
CAMERA_ID = 1
CAMERA_FRAME_SIZE = (1080, 1920)
CAMERA_INITIAL_FOCUS = 40
CALIBRATION_FILE = pathlib.Path("calibration_data.json")
QUIT_KEY = "q"


def main():
    # Загрузка сохраненных данных
    with open(CALIBRATION_FILE) as f:
        data = json.load(f)

    ref_rvec = np.array(data["reference_rvec"])
    ref_tvec = np.array(data["reference_tvec"])
    normal = np.array(data["plane_normal"])
    d = data["plane_d"]
    camera_matrix = np.array(data["camera_matrix"])
    dist_coeffs = np.array(data["distortion_coeffs"])

    # Инициализация камеры
    cam = liara.CameraLab(CAMERA_ID, CAMERA_FRAME_SIZE)
    cam.focus = CAMERA_INITIAL_FOCUS

    click_positions = []
    mouse_pos = (0, 0)

    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_pos, click_positions
        mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            click_positions.append((x, y))

    cv2.namedWindow("Projection Viewer")
    cv2.setMouseCallback("Projection Viewer", mouse_callback)

    print("Кликайте левой кнопкой мыши. Нажмите 'q' для выхода")

    while True:
        frame = cam.get_frame()

        # Отображение позиции мыши
        cv2.putText(frame, f"Mouse: {mouse_pos[0]}, {mouse_pos[1]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Обработка кликов
        for click in click_positions:
            u, v = click

            # 1. Убираем дисторсию
            pts = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float32),
                                      camera_matrix, dist_coeffs)
            pt_undist = pts[0][0]

            # 2. Создаем луч в 3D пространстве камеры
            ray_dir = np.array([pt_undist[0], pt_undist[1], 1.0])
            ray_dir = ray_dir / np.linalg.norm(ray_dir)

            # 3. Переводим в мировую систему координат
            rotation_matrix, _ = cv2.Rodrigues(ref_rvec)
            world_ray_dir = rotation_matrix.T @ ray_dir
            world_ray_origin = rotation_matrix.T @ -ref_tvec[0]

            # 4. Находим пересечение с плоскостью
            denom = np.dot(normal, world_ray_dir)
            if abs(denom) > 1e-6:
                t = -(np.dot(normal, world_ray_origin) + d) / denom
                world_point = world_ray_origin + t * world_ray_dir

                # Рисуем точку и подпись
                cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
                coord_text = f"({world_point[0]:.3f}, {world_point[1]:.3f}, {world_point[2]:.3f})"
                cv2.putText(frame, coord_text, (u + 10, v),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Projection Viewer", cv2.resize(frame, (1920, 1080)))
        key = cv2.waitKey(1) & 0xFF
        if key == ord(QUIT_KEY):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()