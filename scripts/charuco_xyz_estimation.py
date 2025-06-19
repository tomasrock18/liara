import cv2
import numpy


# TODO: Это надо доделать!
def pixel_to_world(pixel_coords, intrinsic_matrix, rotation_vector, translation_vector):
    """
    Преобразует пиксельные координаты в мировые координаты

    Аргументы:
    pixel_coords -- (x, y) координаты пикселя
    intrinsic_matrix -- матрица внутренних параметров камеры (3x3)
    rotation_vector -- вектор поворота (3x1)
    translation_vector -- вектор смещения (3x1)

    Возвращает:
    (x, y, z) мировые координаты
    """
    # Преобразуем вектор поворота в матрицу поворота
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Создаем матрицу проекции [R|t]
    rt_matrix = numpy.hstack((rotation_matrix, translation_vector))

    # Получаем матрицу проекции P = K[R|t]
    projection_matrix = intrinsic_matrix @ rt_matrix

    # Преобразуем пиксельные координаты в однородные координаты
    pixel_homogeneous = numpy.array([pixel_coords[0], pixel_coords[1], 1])

    # Вычисляем мировые координаты (решаем систему уравнений)
    # P * [X,Y,Z,1]^T = s * [u,v,1]^T
    # Мы можем решить это как систему линейных уравнений

    # Выбираем 2 уравнения из 3 (обычно первые два)
    A = projection_matrix[:2, :3] - pixel_homogeneous[:2, numpy.newaxis] * projection_matrix[2, :3]
    b = (pixel_homogeneous[:2] * projection_matrix[2, 3] - projection_matrix[:2, 3])

    # Решаем систему (используем SVD для устойчивости)
    _, _, V = numpy.linalg.svd(A)
    world_coords = V[-1, :]

    # Нормализуем координаты
    world_coords = world_coords / world_coords[-1]

    return world_coords[:3]
