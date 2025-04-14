import liara

import cv2
import flask

app: flask = flask.Flask("Liara")
camera: liara.Camera = liara.Camera()
board: liara.Board = liara.Board()
calibration: liara.Calibration = liara.Calibration()


# Board routes
################################################################################
@app.route("/board/size", methods=["GET", "PUT"])
def board_size() -> flask.Response:
    if flask.request.method == "GET":
        return flask.jsonify(
            {
                "squares_amount_horizontal": board.size[0],
                "squares_amount_vertical": board.size[1],
            }
        )
    board.size = (flask.request.json["squares_amount_horizontal"], flask.request.json["squares_amount_vertical"])
    return flask.Response(status=204)


@app.route("/board/marker-length", methods=["GET", "PUT"])
def board_marker_length() -> flask.Response:
    if flask.request.method == "GET":
        return flask.Response(str(board.marker_length))
    board.marker_length = flask.request.json["marker-length"]
    return flask.Response(status=204)


@app.route("/board/square-length", methods=["GET", "PUT"])
def board_square_length() -> flask.Response:
    if flask.request.method == "GET":
        return flask.Response(str(board.square_length))
    board.square_length = flask.request.json["square-length"]
    return flask.Response(status=204)


@app.route("/board/configure-board-template", methods=["GET", "PUT"])
def board_configure_board_template() -> flask.Response:
    if flask.request.method == "GET":
        return flask.Response(str(board.is_board_initialized), 200)
    board.configure_board_template()
    return flask.Response(status=204)


################################################################################


# Camera routes
################################################################################
@app.route("/camera/camera-id", methods=["GET", "PUT"])
def camera_id() -> flask.Response:
    if flask.request.method == "GET":
        return flask.Response(str(camera.camera_id), status=200)
    camera.camera_id = flask.request.json["camera_id"]
    return flask.Response(status=204)


@app.route("/camera/frame-width", methods=["GET", "PUT"])
def frame_width() -> flask.Response:
    if flask.request.method == "GET":
        return flask.Response(str(camera.frame_width), status=200)
    camera.frame_width = flask.request.json["frame_width"]
    return flask.Response(status=204)


@app.route("/camera/frame-height", methods=["GET", "PUT"])
def frame_height() -> flask.Response:
    if flask.request.method == "GET":
        return flask.Response(str(camera.frame_height), status=200)
    camera.frame_height = flask.request.json["frame_height"]
    return flask.Response(status=204)


@app.route("/camera/codec", methods=["GET", "PUT"])
def codec() -> flask.Response:
    if flask.request.method == "GET":
        return flask.Response(str(camera.codec), status=200)
    camera.codec = flask.request.json["codec"]
    return flask.Response(status=204)


@app.route("/camera/configure-framer", methods=["GET", "PUT"])
def configure_framer() -> flask.Response:
    if flask.request.method == "GET":
        return flask.Response(str(camera.is_framer_configured), status=200)
    camera.configure_framer()
    return flask.Response(status=204)


@app.route("/camera/get-frame", methods=["GET"])
def get_frame() -> flask.Response:
    if flask.request.method == "GET":
        buffer = cv2.imencode(".jpg", camera.get_frame())[1]
        return flask.Response(buffer.tobytes(), status=200, mimetype="image/jpeg")


################################################################################

# Calibration routes
################################################################################
@app.route("/calibration/calibrate", methods=["PUT"])
def calibrate() -> flask.Response:
    if flask.request.method == "PUT":
        calibration.calibrate()
        return flask.Response(status=204)


@app.route("/calibration/is-calibration-possible", methods=["GET"])
def is_calibration_possible() -> flask.Response:
    if flask.request.method == "GET":
        return flask.Response(str(calibration.is_calibration_possible), status=200)


@app.route("/calibration/add-image", methods=["PUT"])
def add_image() -> flask.Response:
    if flask.request.method == "PUT":
        calibration.add_image(flask.request.json["image"])
        return flask.Response(status=204)


@app.route("/calibration/image/<id>", methods=["GET"])
def get_image(idx: str) -> flask.Response:
    if flask.request.method == "GET":
        buffer = cv2.imencode(".jpg", calibration.get_image(int(idx)))[1]
        return flask.Response(buffer.tobytes(), status=200, mimetype="image/jpeg")


@app.route("/calibration/image-with-aruco/<id>", methods=["GET"])
def get_image_with_aruco(idx: str) -> flask.Response:
    if flask.request.method == "GET":
        buffer = cv2.imencode(".jpg", calibration.get_image_with_markers(int(idx)))[1]
        return flask.Response(buffer.tobytes(), status=200, mimetype="image/jpeg")


@app.route("/calibration/image-with-corners/<id>", methods=["GET"])
def get_image_with_corners(idx: str) -> flask.Response:
    if flask.request.method == "GET":
        buffer = cv2.imencode(".jpg", calibration.get_image_with_corners(int(idx)))[1]
        return flask.Response(buffer.tobytes(), status=200, mimetype="image/jpeg")


@app.route("/calibration/clear", methods=["PUT"])
def clear() -> flask.Response:
    if flask.request.method == "PUT":
        calibration.clear_images()
        return flask.Response(status=204)




################################################################################


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5005,
    )
