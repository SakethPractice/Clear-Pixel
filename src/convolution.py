from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_KERNEL_SIZE = 3


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


def build_blur_kernel(kernel_size: int) -> np.ndarray:
    return np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)


def build_sharpen_kernel(kernel_size: int) -> np.ndarray:
    kernel = -1 * np.ones((kernel_size, kernel_size), dtype=np.float32)
    center_index = kernel_size // 2
    kernel[center_index, center_index] = (kernel_size * kernel_size) + (kernel_size * kernel_size - 2)
    return kernel


def get_kernel(mode: str, kernel_size: int = DEFAULT_KERNEL_SIZE) -> np.ndarray:
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("Kernel size must be an odd number greater than or equal to 3.")

    kernels = {
        "blur": build_blur_kernel(kernel_size),
        "sharpen": build_sharpen_kernel(kernel_size),
    }
    return kernels[mode]


def process_image(image: np.ndarray, mode: str, kernel_size: int = DEFAULT_KERNEL_SIZE) -> tuple[np.ndarray, np.ndarray]:
    kernel = get_kernel(mode, kernel_size)
    processed_image = convolve_matrices(image, kernel)
    return processed_image, kernel


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
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=DEFAULT_KERNEL_SIZE,
        help="Odd kernel size to use for convolution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    image = load_image(input_path)
    processed_image, kernel = process_image(image, args.mode, args.kernel_size)
    save_image(output_path, processed_image)

    print(f"Mode applied : {args.mode}")
    print(f"Kernel size  : {args.kernel_size}")
    print(f"Input image  : {input_path}")
    print(f"Output image : {output_path}")
    print("Kernel used:")
    print(kernel)


if __name__ == "__main__":
    main()
