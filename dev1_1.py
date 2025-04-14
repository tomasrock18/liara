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


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Детекция маркеров
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