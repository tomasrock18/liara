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
    if None not in (charuco_corners, charuco_ids):
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
    if cv2.waitKey(1) & 0xFF == ord("a"):
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
    elif cv2.waitKey(1) & 0xFF == ord("\r"):
        projection_error_calibration, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners,
            all_charuco_ids,
            calibration_board,
            frame.shape,
            cameraMatrix=None,
            distCoeffs=None,
        )
        cv2.destroyAllWindows()
        break

#####################################################################
