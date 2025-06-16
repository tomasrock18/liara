import cv2
import liara  # твой пакет с camera, detector, utils

if __name__ == "__main__":
    # 1. Камера
    cam = liara.CameraLab(1, (1920, 1080))

    # 2. Фон
    bg = liara.extract_background(cam)

    # 3. Эталонный контур
    master_contour = liara.extract_master_contour(cam, bg)

    # 4. Калибровка
    calibration_matrix = [
        [1434, 0, 940],
        [0, 1438, 533],
        [0, 0, 1]
    ]

    # 5. Высота камеры (в мм)
    camera_height = 42 - 4.4

    # 6. RT-матрица (если не используешь — можно заменить на np.eye(4))
    rt_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]

    # 7. Запуск визуальной настройки
    liara.start_detector_tuning(
        cam=cam,
        calibration_matrix=calibration_matrix,
        rt_matrix=rt_matrix,
        bg=bg,
        master_contour=master_contour,
        camera_height=camera_height
    )
