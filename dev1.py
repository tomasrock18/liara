from flask import Flask, render_template, Response
import cv2
import numpy

app = Flask(__name__)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# зададим калибровочную матрицу
K = numpy.asarray(
    [
        [1434, 0, 940],
        [0, 1438, 533],
        [0, 0, 1]
    ]
)
K_inv = numpy.linalg.inv(K)
z = 45


def generate_frames():
    while True:

        ## read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # TODO: Обработка фрейма
            corners, ids, _ = cv2.aruco.detectMarkers(
                frame,
                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
            )
            # if len(corners) > 0:
            #     print(corners[0][0][0])

            if len(corners) > 0:
                uv1 = numpy.array([*corners[0][0][0], 1.0])
                # print(uv1)
                xyz = z * (K_inv @ uv1)
                # print(xyz)
                # cv2.putText(frame, f"{xyz}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, 2)

            # cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
