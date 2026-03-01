import cv2
import numpy as np

from app.operators.blurring.blur import Blur


def test_blur_uses_width_height_kernel_order() -> None:
    image = np.arange(8 * 9, dtype=np.uint8).reshape(8, 9)

    out = Blur({"widthSize": 3, "heightSize": 5, "pointX": -1, "pointY": -1}).compute(image)
    expected = cv2.blur(image, (3, 5), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)

    assert np.array_equal(out, expected)
