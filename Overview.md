# Clear-Pixel: Mathematical Overview

This document explains the math behind **Clear-Pixel** in a clean, presentation-friendly way. The project works by treating an image as a matrix of numbers and then applying **convolution** using a small matrix called a **kernel**.

## 1. Core idea

An image is really a grid of pixel values.

- In a grayscale image, each pixel has one number.
- In a color image, each pixel has three numbers:
  - Red
  - Green
  - Blue

So if we write `I(x, y)`, that means:

- `x` = horizontal position
- `y` = vertical position
- `I(x, y)` = pixel value at that location

For a color image, we can think of:

`I(x, y) = [R(x, y), G(x, y), B(x, y)]`

The app applies a small matrix called a kernel to the image. This kernel slides across the image and changes each pixel based on its neighbors.

That process is called **convolution**.

## 2. General convolution formula

If:

- `I` is the input image
- `K` is the kernel
- `O` is the output image

then each output pixel is computed using nearby input pixels.

A clean way to say it is:

`O(x, y) = sum of [kernel value x nearby pixel value] over the whole kernel`

More explicitly:

`O(x, y) = sum(i = -r to r) sum(j = -r to r) K(i, j) * I(x - i, y - j)`

where:

- `k` = kernel size
- `r = (k - 1) / 2`

Why does the kernel size need to be odd?

- A `3 x 3` kernel has a clear center
- A `5 x 5` kernel has a clear center
- A `4 x 4` kernel does not

That center is important because the new output value is assigned to the center position of the kernel.

## 3. What convolution means in simple words

For every pixel:

1. Place the kernel over that pixel.
2. Multiply each kernel entry by the matching image pixel.
3. Add all those products.
4. Store the result as the new pixel value.
5. Repeat for the full image.

This is the main mathematical operation used in Clear-Pixel.

## 4. Blur mode

The blur filter uses an **averaging kernel**.

For a kernel of size `k x k`, the blur kernel is:

- every entry is `1 / (k x k)`
- the sum of all kernel values is `1`

That means all nearby pixels contribute equally.

### Example: 3 x 3 blur kernel

```text
1/9  1/9  1/9
1/9  1/9  1/9
1/9  1/9  1/9
```

So each output pixel becomes:

`O(x, y) = average of the 9 neighboring pixels`

More explicitly:

`O(x, y) = (1/9) * [sum of all values in the 3 x 3 neighborhood]`

### Why blur works

Blur works because averaging reduces differences between neighboring pixels.

- Bright and dark regions mix together
- Sharp transitions become softer
- Fine detail is reduced
- Random noise may also be reduced

That is why blur is often called a **smoothing filter** or a **low-pass filter**.

## 5. Blur example with numbers

Suppose the 3 x 3 neighborhood around a pixel is:

```text
100  110  120
 90  100  110
 80   90  100
```

Then the blur output is:

`O(x, y) = (1/9) * (100 + 110 + 120 + 90 + 100 + 110 + 80 + 90 + 100)`

`O(x, y) = (1/9) * 900`

`O(x, y) = 100`

So the new center pixel becomes `100`, which is the local average.

## 6. Sharpen mode

The sharpen filter in this project is custom-built.

The code does this:

1. Fill the whole kernel with `-1`
2. Replace the center value with:

`2k^2 - 2`

So the sharpen kernel has:

- `-1` everywhere around the center
- a large positive value at the center

This strongly emphasizes the center pixel while subtracting neighboring values.

### Example: 3 x 3 sharpen kernel

For `k = 3`:

`2k^2 - 2 = 2(9) - 2 = 16`

So the kernel becomes:

```text
-1  -1  -1
-1  16  -1
-1  -1  -1
```

### Why sharpening works

This kernel boosts the center pixel and subtracts nearby pixels.

That increases local contrast, especially at edges.

Visual result:

- edges look stronger
- fine detail becomes clearer
- transitions between dark and bright regions stand out more
- too much sharpening can also amplify noise

## 7. Sharpen example with numbers

Using the same neighborhood:

```text
100  110  120
 90  100  110
 80   90  100
```

Apply the sharpen kernel:

```text
-1  -1  -1
-1  16  -1
-1  -1  -1
```

Then:

`O(x, y) = (-1)(100) + (-1)(110) + (-1)(120) + (-1)(90) + (16)(100) + (-1)(110) + (-1)(80) + (-1)(90) + (-1)(100)`

`O(x, y) = 1600 - 800 = 800`

This is much larger than a normal pixel value.

In real image processing, pixel values must stay in the range:

`0 to 255`

So values that become too large are clipped back into the valid range.

That is why sharpening can make bright edges appear very strong.

## 8. Why kernel size changes the effect

The app allows odd kernel sizes from `3` to `15`.

### In blur mode

A larger blur kernel means:

- more nearby pixels are included
- the average is taken over a wider area
- the image becomes softer

So larger blur kernels create stronger smoothing.

### In sharpen mode

The center value grows as:

`2k^2 - 2`

Examples:

- for `k = 3`, center = `16`
- for `k = 5`, center = `48`
- for `k = 7`, center = `96`

So as the kernel gets larger, the sharpening effect becomes much stronger.

## 9. Sum of kernel values

This is useful to mention in a presentation.

### Blur kernel sum

For blur:

- each entry is `1 / (k x k)`
- there are `k x k` entries

So the total sum is:

`1`

This helps preserve overall brightness.

### Sharpen kernel sum

For sharpen:

- there are `(k^2 - 1)` entries equal to `-1`
- the center is `2k^2 - 2`

So the total sum is:

`(2k^2 - 2) - (k^2 - 1) = k^2 - 1`

For `k = 3`, that becomes:

`16 - 8 = 8`

This means the sharpen kernel is **not normalized**. It strongly boosts the center compared with the surrounding pixels.

## 10. Color image processing

The uploaded image is a color image, so convolution is applied separately to each channel:

- Blue
- Green
- Red

That means:

- output blue = kernel applied to blue channel
- output green = kernel applied to green channel
- output red = kernel applied to red channel

Then the channels are combined back into the final image.

So the color structure is preserved while the blur or sharpen effect is applied.

## 11. Border handling

One practical issue is the image boundary.

For example:

- the top-left corner does not have pixels above or to the left
- the bottom-right corner does not have pixels beyond the image edge

But convolution still needs neighboring values.

OpenCV handles this automatically using built-in border rules, so even edge pixels can be processed correctly.

## 12. Signal-processing interpretation

This is a nice higher-level explanation for presentations.

### Blur

Blur removes rapid changes in intensity.

So it reduces:

- noise
- fine texture
- sharp boundaries

That is why it behaves like a low-pass filter.

### Sharpen

Sharpen increases local differences.

So it makes:

- edges stronger
- details clearer
- contrast around boundaries more visible

## 13. Key formulas to remember

- General convolution:
  - `O(x, y) = sum(i = -r to r) sum(j = -r to r) K(i, j) * I(x - i, y - j)`
- Blur kernel:
  - every value = `1 / k^2`
- Sharpen kernel center:
  - `2k^2 - 2`
- Blur kernel sum:
  - `1`
- Sharpen kernel sum:
  - `k^2 - 1`
- Approximate runtime:
  - `O(H x W x k^2)`