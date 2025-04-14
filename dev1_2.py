from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Калибровочная матрица
K = np.array([
    [1434, 0, 940],
    [0, 1438, 533],
    [0, 0, 1]
])
K_inv = np.linalg.inv(K)
z = 45  # Фиксированное значение глубины


def project_3d_to_2d(point_3d, K):
    """Проецирует 3D точку в 2D пиксельные координаты"""
    point_2d = K @ point_3d
    point_2d = point_2d / point_2d[2]  # Нормализация
    return (int(point_2d[0]), int(point_2d[1]))


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 1. Всегда отображаем центральную точку (0,0,45)
            center_3d = np.array([[0], [0], [z]])
            center_2d = project_3d_to_2d(center_3d, K)

            # Рисуем красную точку
            cv2.circle(frame, center_2d, 8, (0, 0, 255), -1)

            # Подписываем точку
            # cv2.putText(frame, "Center (0,0,45)",
            #             (center_2d[0] + 15, center_2d[1] + 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            #             (0, 0, 255), 2, cv2.LINE_AA)

            # 2. Детекция маркеров
            corners, ids, _ = cv2.aruco.detectMarkers(
                frame,
                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
            )

            if ids is not None:
                # Отрисовка маркеров
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # Для каждого обнаруженного маркера
                for i in range(len(ids)):
                    # Для каждого угла маркера
                    for j in range(4):
                        # Получаем координаты угла
                        corner_x = int(corners[i][0][j][0])
                        corner_y = int(corners[i][0][j][1])

                        # Преобразуем в однородные координаты
                        uv = np.array([[corner_x], [corner_y], [1]])

                        # Вычисляем 3D координаты
                        xyz = z * (K_inv @ uv)
                        x, y = xyz[0][0], xyz[1][0]

                        # Форматируем текст координат
                        coord_text = f"({x:.1f}, {y:.1f}, {z})"

                        # Позиция для текста (смещение от угла)
                        text_pos = (corner_x + 10, corner_y + 10)

                        # Отрисовываем координаты
                        cv2.putText(
                            frame,
                            coord_text,
                            text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255, 0, 0),  # Зеленый цвет
                            1,
                            cv2.LINE_AA
                        )

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)