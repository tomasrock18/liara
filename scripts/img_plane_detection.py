import json
import cv2
import numpy as np
import pathlib
import liara

# Параметры
CAMERA_ID = 1
CAMERA_FRAME_SIZE = (1080, 1920)
CAMERA_INITIAL_FOCUS = 40
TARGET_IDS = {3, 6, 24, 27}
BOARD_MARKER_LENGTH_M = 0.025
SAVE_KEY = "s"
QUIT_KEY = "q"
CALIBRATION_FILE = pathlib.Path("calibration_data.json")

# Глобальные переменные
reference_rvec = None
reference_tvec = None

def main():
    global reference_rvec, reference_tvec

    # Загрузка калибровки камеры
    with open("calibration_results.json") as f:
        calib = json.load(f)
    camera_matrix = np.array(calib["intrinsic_matrix"])
    dist_coeffs = np.array(calib["distortion_vector"])

    # Инициализация детектора ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    # Инициализация камеры
    cam = liara.CameraLab(CAMERA_ID, CAMERA_FRAME_SIZE)
    cam.focus = CAMERA_INITIAL_FOCUS

    cv2.namedWindow("Calibration")
    print("Наведите камеру на маркеры. Нажмите 's' для сохранения, 'q' для выхода")

    while True:
        frame = cam.get_frame()
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, BOARD_MARKER_LENGTH_M, camera_matrix, dist_coeffs)

            # Находим опорный маркер (ID=3)
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == 3:
                    reference_rvec = rvecs[i]
                    reference_tvec = tvecs[i]
                    break

            # Если нашли опорный маркер, вычисляем плоскость
            if reference_rvec is not None:
                plane_points = []
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id in TARGET_IDS:
                        # Преобразуем координаты относительно опорного маркера
                        rotation_matrix, _ = cv2.Rodrigues(reference_rvec)
                        relative_tvec = tvecs[i][0] - reference_tvec[0]
                        world_coords = rotation_matrix.T @ relative_tvec
                        plane_points.append(world_coords)

                # Вычисляем уравнение плоскости
                if len(plane_points) >= 3:
                    points = np.array(plane_points)
                    centroid = np.mean(points, axis=0)
                    centered = points - centroid
                    _, _, vh = np.linalg.svd(centered)
                    normal = vh[2, :]
                    d = -np.dot(normal, centroid)

                    # Отображаем уравнение плоскости
                    cv2.putText(frame,
                        f"Plane: {normal[0]:.2f}x + {normal[1]:.2f}y + {normal[2]:.2f}z + {d:.2f} = 0",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Calibration", cv2.resize(frame, (1920, 1080)))
        key = cv2.waitKey(1) & 0xFF

        if key == ord(QUIT_KEY):
            break
        elif key == ord(SAVE_KEY) and reference_rvec is not None:
            # Сохраняем все необходимые данные
            data_to_save = {
                "reference_rvec": reference_rvec.tolist(),
                "reference_tvec": reference_tvec.tolist(),
                "plane_normal": normal.tolist(),
                "plane_d": float(d),
                "camera_matrix": camera_matrix.tolist(),
                "distortion_coeffs": dist_coeffs.tolist()
            }
            with open(CALIBRATION_FILE, "w") as f:
                json.dump(data_to_save, f)
            print(f"Данные сохранены в {CALIBRATION_FILE}")
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()