import cv2
import pathlib

import liara

# Инициализация параметров для калибровки
CAMERA_ID = 1
CAMERA_FRAME_SIZE = (1920, 1080)
BOARD_SIZE = (7, 9)
BOARD_MARKER_LENGTH_M = 0.025
BOARD_SQUARE_LENGTH_M = 0.03
BOARD_ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

# Инициализация параметров для управления псевдоинтерфейсом
INTERFACE_SAVE_BUTTON = "s"
INTERFACE_ADD_BUTTON = "a"
INTERFACE_CALIBRATE_BUTTON = "c"

# Инициализация пути к папке с сохранёнными изображениями
OTHER_IMG_DIR_PATH = pathlib.Path("images")

if __name__ == "__main__":
    # Создание папки с сохранёнными изображениями
    OTHER_IMG_DIR_PATH.mkdir(exist_ok=True)

    # Создание объекта доски ChAruco
    calibration_board = cv2.aruco.CharucoBoard(
        size=BOARD_SIZE,
        markerLength=BOARD_MARKER_LENGTH_M,
        squareLength=BOARD_SQUARE_LENGTH_M,
        dictionary=BOARD_ARUCO_DICTIONARY,
    )

    # Инициализация массива изображений для калибровки
    calibration_images = []

    # Инициализация объекта управления камерой
    cam = liara.CameraLab(
        camera_id=CAMERA_ID,
        frame_size=CAMERA_FRAME_SIZE,
    )

    # Запуск цикла захвата изображений
    try:
        while True:
            # Получение снимка от камеры
            frame = cam.get_frame()

            # Проверка снимка на пригодность к калибровке
            is_img_good = liara.is_good_for_charuco_calibration(
                img=frame,
                charuco_board=calibration_board
            )

            # Обнаружение маркеров Aruco
            corners, ids, _ = cv2.aruco.detectMarkers(
                image=frame,
                dictionary=BOARD_ARUCO_DICTIONARY,
            )

            # Разметка интерфейса
            cv2.putText(
                img=frame,
                text=f"Images: {len(calibration_images)}",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 255),
                thickness=2
            )
            cv2.putText(
                img=frame,
                text=f"Good Image" if is_img_good else "Bad Image",
                org=(10, 60),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0) if is_img_good else (255, 0, 0),
                thickness=2
            )

            # Отрисовка маркеров Aruco
            cv2.aruco.drawDetectedMarkers(
                image=frame,
                corners=corners,
                ids=ids,
            )

            # Вывод окна утилиты
            cv2.imshow("Calibration tool", frame)

            # Обработка пользовательской команды
            key = cv2.waitKey(1) & 0xFF
            if key == ord(INTERFACE_SAVE_BUTTON):
                img_path_string = (OTHER_IMG_DIR_PATH / f"{len(calibration_images)}.png").as_posix()
                cv2.imwrite(img_path_string, frame)
                print(f"Image saved: {img_path_string}")
            elif key == ord(INTERFACE_ADD_BUTTON):
                calibration_images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                print(f"Image added: {len(calibration_images) + 1}")
            elif key == ord(INTERFACE_CALIBRATE_BUTTON):
                print("All images captured! Calibration started!")
                break
    finally:
        cv2.destroyAllWindows()

    # Запуск процесса калибровки
    calibration = liara.CalibrationChAruco(
        images=calibration_images,
        board=calibration_board,
    )

    # Вывод результатов калибровки
    print(f"intrinsic camera parameters: {calibration.intrinsic_parameters()}")
    print(f"Distortion coefficients: {calibration.distortion_coefficients()}")
