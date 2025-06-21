import cv2
import numpy as np

# 1) Объявляем входные данные скрипта
#####################################################################
CAMERA_ID = 1
CAMERA_FRAME_SIZE = (1920, 1080)
BOARD_SIZE = (7, 9)
BOARD_MARKER_LENGTH_M = 0.025
BOARD_SQUARE_LENGTH_M = 0.03
BOARD_ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
PLANE_MARKERS_IDS = (3, 6, 24, 27)
PLANE_ANCHOR_MARKER_ID = 3
MIN_CONTOUR_AREA = 500
SIMILARITY_THRESHOLD = 0.7
#####################################################################

threshold_value = 25
similarity_threshold = 0.7

def setup_camera():
    cam = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_SIZE[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_SIZE[1])
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    return cam

def set_focus_window(cam):
    window_name = "Camera set up"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_focus_change(value):
        cam.set(cv2.CAP_PROP_FOCUS, int(value))

    cv2.createTrackbar("Focus", window_name, int(cam.get(cv2.CAP_PROP_FOCUS)), 255, on_focus_change)

    while True:
        ret, frame = cam.read()
        if not ret:
            raise Exception("Камера не вернула картинку")
        cv2.putText(frame, f"Current focus: {int(cam.get(cv2.CAP_PROP_FOCUS))}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(frame, "Press 'Enter' to continue", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 13:
            cv2.destroyAllWindows()
            break

def calibrate_camera(cam):
    board = cv2.aruco.CharucoBoard(
        size=BOARD_SIZE,
        markerLength=BOARD_MARKER_LENGTH_M,
        squareLength=BOARD_SQUARE_LENGTH_M,
        dictionary=BOARD_ARUCO_DICTIONARY
    )
    detector = cv2.aruco.CharucoDetector(board)
    all_corners = []
    all_ids = []

    window_name = "Initial Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cam.read()
        if not ret:
            raise Exception("Камера не вернула картинку")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)

        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(frame, corners, ids)

        cv2.putText(frame, f"Frames for calibration: {len(all_corners)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(frame, "Press 'a' to add frame for calibration", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Press 'Enter' to start calibration", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("a") and corners is not None:
            all_corners.append(corners)
            all_ids.append(ids)
        elif key == 13 and len(all_corners) > 0:
            _, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
                all_corners, all_ids, board, gray.shape, None, None
            )
            cv2.destroyAllWindows()
            return camera_matrix, dist_coeffs

def define_plane(cam, camera_matrix, dist_coeffs):
    aruco_detector = cv2.aruco.ArucoDetector(BOARD_ARUCO_DICTIONARY, cv2.aruco.DetectorParameters())
    plane_normal = None
    plane_d = None
    anchor_rot_matrix = None
    anchor_tvec = None

    window_name = "Plane Definition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cam.read()
        if not ret:
            raise Exception("Камера не вернула картинку")

        corners, ids, _ = aruco_detector.detectMarkers(frame)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, BOARD_MARKER_LENGTH_M, camera_matrix, dist_coeffs)

        anchor_rvec = None
        anchor_tvec = None
        plane_points = []

        if ids is not None and rvecs is not None:
            for i, mid in enumerate(ids.flatten()):
                if mid == PLANE_ANCHOR_MARKER_ID:
                    anchor_rvec = rvecs[i]
                    anchor_tvec = tvecs[i]
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, anchor_rvec, anchor_tvec, 0.05)

            if anchor_rvec is not None:
                anchor_rot_matrix, _ = cv2.Rodrigues(anchor_rvec)
                for i, mid in enumerate(ids.flatten()):
                    if mid in PLANE_MARKERS_IDS:
                        world_coords = anchor_rot_matrix.T @ (tvecs[i][0] - anchor_tvec[0])
                        plane_points.append(world_coords)

        if len(plane_points) >= 3:
            points = np.array(plane_points)
            centroid = np.mean(points, axis=0)
            centered = points - centroid
            _, _, vh = np.linalg.svd(centered)
            plane_normal = vh[2, :]
            plane_d = -np.dot(plane_normal, centroid)

        text_pos = (10, 30)
        if plane_normal is not None:
            eq = f"Plane: {plane_normal[0]:.2f}x + {plane_normal[1]:.2f}y + {plane_normal[2]:.2f}z + {plane_d:.2f} = 0"
            cv2.putText(frame, eq, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Not enough markers to build a plane", text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "Press 'Enter' to continue", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 13:
            cv2.destroyAllWindows()
            break

    return plane_normal, plane_d, anchor_rot_matrix, anchor_tvec

def get_background(cam, num_frames=10):
    print("Capturing background...")
    frames = []
    for _ in range(num_frames):
        ret, frame = cam.read()
        if ret:
            frames.append(frame)
        cv2.waitKey(50)
    bg = np.median(frames, axis=0).astype(np.uint8)
    return bg

def detect_template_contour(cam, bg):
    template_contour = None
    threshold_value = 25
    min_area_value = MIN_CONTOUR_AREA
    roi_top = 0
    roi_bottom = 100
    roi_left = 0
    roi_right = 100

    window_name = "Object Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Trackbars
    cv2.createTrackbar('Threshold', window_name, threshold_value, 255, lambda x: None)
    cv2.createTrackbar('Min Area', window_name, min_area_value, 5000, lambda x: None)
    cv2.createTrackbar('ROI Top', window_name, roi_top, 100, lambda x: None)
    cv2.createTrackbar('ROI Bottom', window_name, roi_bottom, 100, lambda x: None)
    cv2.createTrackbar('ROI Left', window_name, roi_left, 100, lambda x: None)
    cv2.createTrackbar('ROI Right', window_name, roi_right, 100, lambda x: None)

    while True:
        ret, frame = cam.read()
        if not ret:
            raise Exception("Камера не вернула картинку")

        height, width = frame.shape[:2]
        top = int(height * cv2.getTrackbarPos('ROI Top', window_name) / 100)
        bottom = int(height * cv2.getTrackbarPos('ROI Bottom', window_name) / 100)
        left = int(width * cv2.getTrackbarPos('ROI Left', window_name) / 100)
        right = int(width * cv2.getTrackbarPos('ROI Right', window_name) / 100)

        roi_frame = frame[top:bottom, left:right]
        roi_bg = bg[top:bottom, left:right]

        gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray_bg = cv2.cvtColor(roi_bg, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_bg, gray_frame)
        thresh_val = cv2.getTrackbarPos('Threshold', window_name)
        _, thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = frame.copy()

        cv2.rectangle(result, (left, top), (right, bottom), (255, 255, 0), 2)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cv2.getTrackbarPos('Min Area', window_name):
                continue
            offset_cnt = cnt + (left, top)
            if template_contour is None:
                template_contour = offset_cnt
                M = cv2.moments(offset_cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(result, (cX, cY), 5, (0, 0, 255), -1)
                    cv2.putText(result, "Template saved!", (cX - 50, cY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.drawContours(result, [offset_cnt], -1, (0, 255, 0), 2)

        cv2.imshow(window_name, result)
        if cv2.waitKey(1) & 0xFF == 13:
            cv2.destroyAllWindows()
            break

    return template_contour

def detect_objects(cam, bg, camera_matrix, plane_normal, plane_d, anchor_rot_matrix, anchor_tvec, template_contour):
    window_name = "3D Coordinates Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Глобальные переменные для trackbar'ов
    global threshold_value, similarity_threshold
    threshold_value = 25
    similarity_threshold = 0.7

    # Создаем трекбары
    cv2.createTrackbar('Threshold', window_name, threshold_value, 255, lambda x: None)
    cv2.createTrackbar('Similarity Threshold', window_name, int(similarity_threshold * 100), 100,
                       lambda x: None)  # масштабируем до 0-100

    while True:
        ret, frame = cam.read()
        if not ret:
            raise Exception("Камера не вернула картинку")

        # Получаем текущие значения из трекбаров
        current_threshold = cv2.getTrackbarPos('Threshold', window_name)
        current_similarity_threshold = cv2.getTrackbarPos('Similarity Threshold', window_name) / 100.0

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_bg, gray_frame)
        _, thresh = cv2.threshold(diff, current_threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = frame.copy()

        if template_contour is not None and plane_normal is not None:
            for cnt in contours:
                if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                    continue

                similarity = cv2.matchShapes(template_contour, cnt, cv2.CONTOURS_MATCH_I2, 0)
                if similarity > current_similarity_threshold:
                    continue

                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                point_3d = pixel_to_3d(cX, cY, plane_normal, plane_d, camera_matrix)
                if point_3d is not None and anchor_rot_matrix is not None and anchor_tvec is not None:
                    world_coords = anchor_rot_matrix.T @ (point_3d - anchor_tvec[0])
                    cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
                    cv2.circle(result, (cX, cY), 5, (0, 0, 255), -1)
                    cv2.putText(result,
                                f"X:{world_coords[0]:.1f} Y:{world_coords[1]:.1f} Z:{world_coords[2]:.1f}",
                                (cX - 100, cY - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Отображение информации
        cv2.putText(result, "Detecting similar objects on plane", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(result, "Press 'Enter' to exit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(window_name, result)

        if cv2.waitKey(1) & 0xFF == 13:
            cv2.destroyAllWindows()
            break

def pixel_to_3d(x, y, normal, d, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ray_dir = np.array([(x - cx)/fx, (y - cy)/fy, 1.0])
    ray_dir /= np.linalg.norm(ray_dir)
    denominator = np.dot(normal, ray_dir)
    if abs(denominator) < 1e-6:
        return None
    t = -(np.dot(normal, np.zeros(3)) + d) / denominator
    return t * ray_dir

def main():
    cam = setup_camera()
    set_focus_window(cam)
    camera_matrix, dist_coeffs = calibrate_camera(cam)
    plane_normal, plane_d, anchor_rot_matrix, anchor_tvec = define_plane(cam, camera_matrix, dist_coeffs)
    bg = get_background(cam)
    template_contour = detect_template_contour(cam, bg)
    detect_objects(cam, bg, camera_matrix, plane_normal, plane_d, anchor_rot_matrix, anchor_tvec, template_contour)
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()