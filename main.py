import cv2

import liara
from liara.utils import extract_master_contour

if __name__ == '__main__':
    cam = liara.CameraLab(0, (1920, 1080))

    bg = liara.extract_background(cam)

    cv2.imshow("Extracted Background", bg)
    cv2.waitKey(0)

    cont = extract_master_contour(cam, bg)
    print(cont)
