import cv2
import numpy

# 1) Объявляем входные данные скрипта
#####################################################################
CAMERA_ID = 1
CAMERA_FRAME_SIZE = (1920, 1080)
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
#####################################################################
