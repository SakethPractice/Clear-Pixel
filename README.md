# Clear-Pixel

Clear-Pixel is an interactive image-processing app built with Streamlit, OpenCV, and NumPy. It demonstrates matrix convolution through a clean visual interface where you can upload an image, apply blur or sharpen filters, compare the original and processed versions, and download the final result.

## Features

- Upload PNG, JPG, or JPEG images from the sidebar
- Apply `blur` or `sharpen` convolution modes
- Adjust the kernel size from `3 x 3` up to `15 x 15`
- Compare original and processed images with an interactive before/after slider
- View processing stats such as operation, kernel size, runtime, and image resolution
- Inspect the kernel used for the transformation
- Download the processed image as a PNG

## Tech Stack

- Python
- Streamlit
- OpenCV
- NumPy

## Project Structure

```text
Clear-Pixel/
|-- app.py
|-- README.md
|-- requirements.txt
`-- src/
    `-- convolution.py
```

## Installation

1. Clone or download the project.
2. Open a terminal in the `Clear-Pixel` folder.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

From inside the `Clear-Pixel` folder, start the app with:

```bash
streamlit run app.py
```

After Streamlit starts, open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How to Use

1. Upload an image from the sidebar.
2. Choose a processing style:
   - `Soft Blur` for smoothing
   - `Crisp Sharpen` for stronger edge definition
3. Adjust the kernel intensity using the slider.
4. Review the output metrics.
5. Drag the comparison slider to inspect the result.
6. Download the processed image when you are satisfied.

## Convolution Modes

### Blur

The blur option uses a normalized averaging kernel. Increasing the kernel size makes the image progressively softer.

### Sharpen

The sharpen option uses a custom kernel that emphasizes the center pixel while subtracting neighboring values. Increasing the kernel size makes the sharpening effect stronger.

## Command-Line Usage

The convolution logic can also be used from the command line through `src/convolution.py`.

Example:

```bash
python src/convolution.py --input images/sample.jpg --output images/output.jpg --mode blur --kernel-size 5
```

Available arguments:

- `--input`: path to the source image
- `--output`: path to save the processed image
- `--mode`: `blur` or `sharpen`
- `--kernel-size`: odd integer greater than or equal to 3

## Requirements

The app depends on:

```text
numpy
opencv-python
streamlit
```

## Notes

- Kernel size must be an odd number and at least `3`
- Images are exported as `.png` from the Streamlit interface
- The preview is resized only for display; processing is performed on the uploaded image itself

## Future Improvements

- Add more convolution filters such as edge detection or emboss
- Support side-by-side histogram comparison
- Offer preset examples for quick testing
- Add image enhancement controls beyond convolution

## License

This project is for learning and demonstration purposes. Add a license section here if you plan to distribute it publicly.
