import numpy as np
import pytest

from app.operators.blurring.blur import Blur


def test_blur_kernel_applied_in_correct_direction() -> None:
    """
    Regression for GitHub #122: widthSize must map to kernel width (cols)
    and heightSize to kernel height (rows). Previously they were swapped.
    A (1, 9) kernel spreads vertically but not horizontally.
    """
    image = np.zeros((20, 20), dtype=np.uint8)
    image[10, 10] = 255

    out = Blur({"widthSize": 1, "heightSize": 9, "pointX": -1, "pointY": -1}).compute(image)

    assert out[6, 10] > 0, "expected vertical spread"
    assert out[14, 10] > 0, "expected vertical spread"
    assert out[10, 4] == 0, "unexpected horizontal spread"
    assert out[10, 16] == 0, "unexpected horizontal spread"


def test_blur_square_kernel_preserves_shape() -> None:
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    out = Blur({"widthSize": 3, "heightSize": 3, "pointX": -1, "pointY": -1}).compute(image)
    assert out.shape == image.shape


def test_blur_1x1_kernel_is_noop() -> None:
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    out = Blur({"widthSize": 1, "heightSize": 1, "pointX": -1, "pointY": -1}).compute(image)
    assert np.array_equal(out, image)


@pytest.mark.parametrize(
    ("width_size", "height_size"),
    [
        (0, 3),
        (3, 0),
        (-1, 3),
        (3, -1),
        (0, 0),
    ],
)
def test_blur_rejects_non_positive_kernel_dimensions(width_size: int, height_size: int) -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Blur kernel dimensions must be >= 1"):
        Blur({"widthSize": width_size, "heightSize": height_size}).compute(image)
