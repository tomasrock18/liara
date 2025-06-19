import json
import cv2
import numpy as np
import pathlib
import liara

# Параметры
CAMERA_ID = 1
CAMERA_FRAME_SIZE = (1080, 1920)
CAMERA_INITIAL_FOCUS = 40
CALIBRATION_FILE = pathlib.Path("calibration_data.json")
SAVE_KEY = ord('s')
QUIT_KEY = ord('q')


class ObjectTracker:
    def __init__(self):
        self.load_calibration_data()
        self.background = None
        self.master_contour = None
        self.cam = liara.CameraLab(CAMERA_ID, CAMERA_FRAME_SIZE)
        self.cam.focus = CAMERA_INITIAL_FOCUS
        self.min_contour_area = 500
        self.similarity_threshold = 0.7

    def load_calibration_data(self):
        with open(CALIBRATION_FILE) as f:
            data = json.load(f)

        self.ref_rvec = np.array(data["reference_rvec"])
        self.ref_tvec = np.array(data["reference_tvec"])
        self.normal = np.array(data["plane_normal"])
        self.plane_d = data["plane_d"]
        self.camera_matrix = np.array(data["camera_matrix"])
        self.dist_coeffs = np.array(data["distortion_coeffs"])

    def capture_background(self):
        cv2.namedWindow("Background")
        print("Наведите камеру на фон. Нажмите 's' для сохранения, 'q' для выхода")

        while True:
            frame = self.cam.get_frame()
            cv2.imshow("Background", cv2.resize(frame, (1920, 1080)))

            key = cv2.waitKey(1)
            if key == QUIT_KEY:
                exit()
            elif key == SAVE_KEY:
                self.background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.destroyWindow("Background")
                break

    def capture_master_contour(self):
        cv2.namedWindow("Master Contour")
        print("Поместите объект. Нажмите 's' для сохранения контура, 'q' для выхода")

        while True:
            frame = self.cam.get_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Вычитание фона
            diff = cv2.absdiff(gray, self.background)
            _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Поиск контуров
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Отображение
            display = frame.copy()
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > self.min_contour_area:
                    cv2.drawContours(display, [largest], -1, (0, 255, 0), 2)

            cv2.imshow("Master Contour", cv2.resize(display, (1920, 1080)))

            key = cv2.waitKey(1)
            if key == QUIT_KEY:
                exit()
            elif key == SAVE_KEY and contours:
                self.master_contour = largest
                cv2.destroyWindow("Master Contour")
                break

    def calculate_3d_position(self, point_2d):
        # Убираем дисторсию
        pts = cv2.undistortPoints(np.array([[[point_2d[0], point_2d[1]]]], dtype=np.float32),
                                  self.camera_matrix, self.dist_coeffs)
        pt_undist = pts[0][0]

        # Создаем луч в 3D пространстве камеры
        ray_dir = np.array([pt_undist[0], pt_undist[1], 1.0])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        # Переводим в мировую систему координат
        rotation_matrix, _ = cv2.Rodrigues(self.ref_rvec)
        world_ray_dir = rotation_matrix.T @ ray_dir
        world_ray_origin = rotation_matrix.T @ -self.ref_tvec[0]

        # Находим пересечение с плоскостью
        denom = np.dot(self.normal, world_ray_dir)
        if abs(denom) > 1e-6:
            t = -(np.dot(self.normal, world_ray_origin) + self.plane_d) / denom
            return world_ray_origin + t * world_ray_dir
        return None

    def contour_similarity(self, cnt1, cnt2):
        # Вычисляем схожесть контуров через Hu моменты
        moments1 = cv2.HuMoments(cv2.moments(cnt1)).flatten()
        moments2 = cv2.HuMoments(cv2.moments(cnt2)).flatten()

        # Логарифмическое преобразование для Hu моментов
        moments1 = np.sign(moments1) * np.log(np.abs(moments1) + 1e-10)
        moments2 = np.sign(moments2) * np.log(np.abs(moments2) + 1e-10)

        # Косинусная схожесть
        similarity = np.dot(moments1, moments2) / (np.linalg.norm(moments1) * np.linalg.norm(moments2))
        return similarity

    def run_detection(self):
        cv2.namedWindow("Detection")
        print("Обнаружение объектов. Нажмите 'q' для выхода")

        while True:
            frame = self.cam.get_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Вычитание фона
            diff = cv2.absdiff(gray, self.background)
            _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Поиск контуров
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Отображение
            display = frame.copy()
            for cnt in contours:
                if cv2.contourArea(cnt) > self.min_contour_area:
                    similarity = self.contour_similarity(cnt, self.master_contour)
                    if similarity > self.similarity_threshold:
                        # Рисуем контур
                        cv2.drawContours(display, [cnt], -1, (0, 255, 0), 2)

                        # Вычисляем центр масс
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

                            # Вычисляем 3D координаты
                            world_pos = self.calculate_3d_position((cx, cy))
                            if world_pos is not None:
                                coord_text = f"({world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f})"
                                cv2.putText(display, coord_text, (cx + 10, cy),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow("Detection", cv2.resize(display, (1920, 1080)))

            if cv2.waitKey(1) == QUIT_KEY:
                break

        self.cam.release()
        cv2.destroyAllWindows()

    def run(self):
        self.capture_background()
        self.capture_master_contour()
        self.run_detection()


if __name__ == "__main__":
    tracker = ObjectTracker()
    tracker.run()