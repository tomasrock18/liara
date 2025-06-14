import abc

import cv2
import numpy


class CameraBase(abc.ABC):
    @abc.abstractmethod
    def get_frame(self) -> numpy.ndarray:
        pass


class CameraLab(CameraBase):

    def __init__(self, camera_id: int, frame_size: tuple[int, int], codec: str = "MJPG"):
        self._framer: cv2.VideoCapture = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        self._framer.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[1])
        self._framer.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[0])
        self._framer.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*codec))

    def get_frame(self) -> numpy.ndarray:
        return self._framer.read()[1]
