import cv2
import numpy as np
import pytest

from app.operators.blurring.blur import Blur


@pytest.fixture
def image():
    """A small solid-colour image reused across all Blur tests."""
    return np.full((5, 5, 3), 128, dtype=np.uint8)


class TestBlurValidInput:
    def test_default_params_produce_output(self, image):
        result = Blur({}).compute(image)
        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_explicit_valid_dimensions(self, image):
        result = Blur({"widthSize": 3, "heightSize": 3}).compute(image)
        assert result.shape == image.shape

    def test_even_kernel_sizes_are_accepted(self, image):
        # Plain blur supports any positive integer, including even values
        result = Blur({"widthSize": 2, "heightSize": 4}).compute(image)
        assert result.shape == image.shape

    def test_width_one_is_valid(self, image):
        result = Blur({"widthSize": 1, "heightSize": 1}).compute(image)
        assert result.shape == image.shape

    def test_1x1_kernel_is_noop(self):
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        out = Blur({"widthSize": 1, "heightSize": 1, "pointX": -1, "pointY": -1}).compute(image)
        assert np.array_equal(out, image)

    def test_non_square_kernel_correct_axis(self):
        """Regression test: widthSize=1, heightSize=5 must produce the same result
        as cv2.blur(src, (1, 5)) — width=1 (no horizontal spread), height=5
        (vertical blur).  Before the (height, width) → (width, height) fix the
        tuple was transposed, silently applying the wrong-axis blur.
        """
        src = np.arange(75, dtype=np.uint8).reshape(5, 5, 3)
        result = Blur({"widthSize": 1, "heightSize": 5}).compute(src)
        expected = cv2.blur(src, (1, 5), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
        np.testing.assert_array_equal(result, expected)

    def test_kernel_applied_in_correct_direction(self):
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


class TestBlurInvalidInput:
    @pytest.mark.parametrize("bad_width", [0, -1, -10])
    def test_non_positive_width_raises(self, image, bad_width):
        with pytest.raises(ValueError, match="'widthSize'"):
            Blur({"widthSize": bad_width, "heightSize": 3}).compute(image)

    @pytest.mark.parametrize("bad_height", [0, -1, -10])
    def test_non_positive_height_raises(self, image, bad_height):
        with pytest.raises(ValueError, match="'heightSize'"):
            Blur({"widthSize": 3, "heightSize": bad_height}).compute(image)

    def test_error_message_is_user_friendly(self, image):
        with pytest.raises(ValueError, match="positive integer"):
            Blur({"widthSize": 0}).compute(image)
