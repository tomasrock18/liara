import liara

import cv2

if __name__ == '__main__':
    # Сбор изображений для калибровки
    images = []
    cam = liara.CameraLab(1, (1920, 1080))

    try:
        while True:
            frame = cam.get_frame()
            cv2.putText(
                frame,
                f"Images: {len(images)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )
            corners, ids, _ = cv2.aruco.detectMarkers(
                frame,
                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL),
            )
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            cv2.imshow('Calibration', frame)

            # TODO: Убрать or
            if cv2.waitKey(1000) or 0xFF == 27:
                images.append(frame)
                if len(images) == 20:
                    break
    finally:
        cv2.destroyAllWindows()

    # Инициализация доски
    board = cv2.aruco.CharucoBoard(
        size=(7, 9),
        markerLength=0.025,
        squareLength=0.03,
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    )

    # Калибровка
    cal = liara.CalibrationChAruco(
    [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in images],
        board
    )
    print(cal.get_calibration_matrix())
