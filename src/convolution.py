from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


BLUR_KERNEL = np.ones((3, 3), dtype=np.float32) / 9
SHARPEN_KERNEL = np.array(
    [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    dtype=np.float32,
)


def convolve_matrices(image_matrix: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply 2D convolution between an image matrix and a kernel."""
    return cv2.filter2D(image_matrix, -1, kernel)


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def save_image(output_path: Path, image: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not write image: {output_path}")


def get_kernel(mode: str) -> np.ndarray:
    kernels = {
        "blur": BLUR_KERNEL,
        "sharpen": SHARPEN_KERNEL,
    }
    return kernels[mode]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply matrix convolution for image blurring or sharpening.",
    )
    parser.add_argument(
        "--input",
        default="images/sample.jpg",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output",
        default="images/output.jpg",
        help="Path to save the processed image.",
    )
    parser.add_argument(
        "--mode",
        choices=("blur", "sharpen"),
        default="blur",
        help="Convolution mode to apply.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    image = load_image(input_path)
    kernel = get_kernel(args.mode)
    processed_image = convolve_matrices(image, kernel)
    save_image(output_path, processed_image)

    print(f"Mode applied : {args.mode}")
    print(f"Input image  : {input_path}")
    print(f"Output image : {output_path}")
    print("Kernel used:")
    print(kernel)


if __name__ == "__main__":
    main()
