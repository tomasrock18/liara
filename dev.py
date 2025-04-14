import pathlib
import cv2
import numpy

# Определить координаты на изображении
coordinates = []
markers_img_dir_path = pathlib.Path().cwd() / "wip" / "markers"
for marker_img_path in markers_img_dir_path.iterdir():
    marker_img = cv2.imread(marker_img_path)
    corners, ids, _ = cv2.aruco.detectMarkers(marker_img,
                                              cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL))
    for marker_corners in corners:
        # marker_corners: numpy array формы (1, 4, 2), извлекаем и превращаем в (4, 2)
        marker_corners = marker_corners.reshape(-1, 2)

        # Пример: сохранить координаты всех четырёх углов
        coordinates.append(marker_corners)

        # Для примера можно вывести
        print(f"{marker_img_path.name} — координаты углов:")
        for i, (u, v) in enumerate(marker_corners):
            print(f"  Угол {i + 1}: u={u:.1f}, v={v:.1f}")

# зададим калибровочную матрицу
K = numpy.asarray(
    [
        [1434, 0, 940],
        [0, 1438, 533],
        [0, 0, 1]
    ]
)

# зададим расстояние от камеры до плоскости (в мм)
z = 50

# Инвертируем матрицу камеры
K_inv = numpy.linalg.inv(K)

# Сюда будем складывать координаты в системе камеры
points_3d_all = []

for marker_corners in coordinates:
    points_3d = []
    for (u, v) in marker_corners:
        uv1 = numpy.array([u, v, 1.0])
        xyz = z * (K_inv @ uv1)  # получаем [X, Y, Z]
        points_3d.append(xyz)

    points_3d = numpy.array(points_3d)
    points_3d_all.append(points_3d)

    print("3D координаты углов маркера в системе камеры:")
    for i, (x, y, z_) in enumerate(points_3d):
        print(f"  Угол {i + 1}: X={x:.1f} мм, Y={y:.1f} мм, Z={z_:.1f} мм")


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 8))

for marker_id, points_3d in enumerate(points_3d_all):
    xs, ys = points_3d[:, 0], points_3d[:, 1]
    plt.scatter(xs, ys, label=f'Marker {marker_id}', s=40)
    for i, (x, y) in enumerate(zip(xs, ys)):
        plt.text(x, y, f'{i+1}', fontsize=9, ha='center', va='center')

# Настройки осей
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("X (мм)")
plt.ylabel("Y (мм)")
plt.title("Положение углов маркеров на плоскости (XY)")
# plt.legend()
plt.tight_layout()
plt.show()