import cv2


class Board:
    ARUCO_DICTIONARY: cv2.aruco.Dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

    def __init__(self):
        self._board: cv2.aruco.CharucoBoard | None = None
        self._size: tuple[int, int] = (7, 9)
        self._marker_length: float = 0.025
        self._square_length: float = 0.03

    def configure_board_template(self) -> None:
        self._board = cv2.aruco.CharucoBoard(
            size=self._size,
            markerLength=self._marker_length,
            squareLength=self._square_length,
            dictionary=self.ARUCO_DICTIONARY
        )

    def get_board_template(self) -> cv2.aruco.CharucoBoard | None:
        return self._board

    @property
    def is_board_initialized(self) -> bool:
        return self._board is not None

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @size.setter
    def size(self, size: tuple[int, int]) -> None:
        assert len(size) == 2
        self._size = size

    @property
    def marker_length(self) -> float:
        return self._marker_length

    @marker_length.setter
    def marker_length(self, marker_length: float) -> None:
        self._marker_length = marker_length

    @property
    def square_length(self) -> float:
        return self._square_length

    @square_length.setter
    def square_length(self, square_length: float) -> None:
        self._square_length = square_length
