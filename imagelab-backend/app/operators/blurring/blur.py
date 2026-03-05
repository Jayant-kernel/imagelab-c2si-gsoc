import cv2
import numpy as np

from app.operators.base import BaseOperator


class Blur(BaseOperator):
    def compute(self, image: np.ndarray) -> np.ndarray:
        width_size = int(self.params.get("widthSize", 3))
        height_size = int(self.params.get("heightSize", 3))
        point_x = int(self.params.get("pointX", -1))
        point_y = int(self.params.get("pointY", -1))

        if width_size < 1 or height_size < 1:
            raise ValueError(f"Blur kernel dimensions must be >= 1, got width={width_size}, height={height_size}")

        return cv2.blur(
            image,
            (width_size, height_size),
            anchor=(point_x, point_y),
            borderType=cv2.BORDER_DEFAULT,
        )
