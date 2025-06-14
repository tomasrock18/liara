import cv2
import numpy

from .camera import CameraBase


def extract_background(cam: CameraBase) -> numpy.ndarray:
    frame = None
    try:
        while True:
            frame = cam.get_frame()
            cv2.imshow("Background Extractor", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cv2.destroyAllWindows()
        return frame


def extract_master_contour(cam: CameraBase, bg: numpy.ndarray) -> numpy.ndarray:
    window_name = "Contour Extractor"
    cv2.namedWindow(window_name)

    def nothing(*args):
        pass

    # Создание ползунка для порога
    cv2.createTrackbar("Threshold", window_name, 50, 255, nothing)

    contour_result = None

    try:
        while True:
            frame = cam.get_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Вычитание фона
            diff = cv2.absdiff(gray, cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY))

            # Порог от ползунка
            threshold_val = cv2.getTrackbarPos("Threshold", window_name)
            _, thresh = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)

            # Морфологическая очистка
            kernel = numpy.ones((5, 5), numpy.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # Поиск контуров
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Копия для отображения
            display = frame.copy()
            cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

            # Надпись: количество контуров
            cv2.putText(display,
                        f"Detected contours: {len(contours)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2)

            # Показываем
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        if contours:
            # Выбираем самый большой
            contour_result = max(contours, key=cv2.contourArea)
            contour_result = contour_result.reshape(-1, 2)

        return contour_result

    finally:
        cv2.destroyWindow(window_name)
