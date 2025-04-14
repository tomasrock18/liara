import cv2
import numpy


class Camera:
    allowed_codec_strings: list[str] = ["MJPG"]

    def __init__(self):
        self._framer: cv2.VideoCapture | None = None
        self._frame_width: int = 1920
        self._frame_height: int = 1080
        self._framer_codec: str = "MJPG"
        self._camera_id: int = 0

    def configure_framer(self) -> None:
        self._framer = cv2.VideoCapture(self._camera_id, cv2.CAP_DSHOW)
        self._framer.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        self._framer.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)
        self._framer.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*self.codec))

    def get_frame(self) -> numpy.ndarray:
        return self._framer.read()[1]

    @property
    def is_framer_configured(self) -> bool:
        return self._framer is not None

    @property
    def camera_id(self) -> int | None:
        return self._camera_id

    @camera_id.setter
    def camera_id(self, camera_id: int) -> None:
        assert isinstance(camera_id, int), "Camera ID must be an integer."
        self._camera_id = camera_id

    @property
    def frame_width(self) -> int | None:
        return self._frame_width

    @frame_width.setter
    def frame_width(self, frame_width: int) -> None:
        assert isinstance(frame_width, int), "Frame width must be an integer."
        self._frame_width = frame_width

    @property
    def frame_height(self) -> int | None:
        return self._frame_height

    @frame_height.setter
    def frame_height(self, frame_height: int) -> None:
        assert isinstance(frame_height, int), "Frame height must be an integer."
        self._frame_height = frame_height

    @property
    def codec(self) -> str:
        return self._framer_codec

    @codec.setter
    def codec(self, codec: str) -> None:
        if codec in self.allowed_codec_strings:
            raise ValueError(f"{codec} is not a valid codec string.")
        self._framer_codec = codec
