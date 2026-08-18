import cv2
import numpy as np
import os


# ============================================================
# FROM-SCRATCH CORRELATION
# ============================================================

def correlate2d(image, kernel):

    h, w = image.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="reflect"
    )

    output = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):

            region = padded[
                i:i+kh,
                j:j+kw
            ]

            output[i, j] = np.sum(
                region * kernel
            )

    return output


# ============================================================
# FROM-SCRATCH CONVOLUTION
# ============================================================

def convolve2d(image, kernel):

    # Flip kernel both vertically and horizontally
    flipped_kernel = np.flipud(
        np.fliplr(kernel)
    )

    return correlate2d(
        image,
        flipped_kernel
    )


# ============================================================
# LAPLACIAN SHARPENING
# ============================================================

def laplacian_sharpen(image, kernel, operation):

    if operation == "correlation":

        laplacian = correlate2d(
            image,
            kernel
        )

    else:

        laplacian = convolve2d(
            image,
            kernel
        )

    image_float = image.astype(
        np.float32
    )

    # Sharpening:
    # g = f + Laplacian
    sharpened = image_float + laplacian

    return np.clip(
        sharpened,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# LOAD IMAGE
# ============================================================

image_path = "../dataset/agv_motion_blur.png"

image = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        "Could not load agv_motion_blur.png"
    )


# ============================================================
# 4-NEIGHBOR LAPLACIAN
# ============================================================

laplacian_4 = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=np.float32)


# ============================================================
# CORRELATION
# ============================================================

result_correlation = laplacian_sharpen(
    image,
    laplacian_4,
    "correlation"
)


# ============================================================
# CONVOLUTION
# ============================================================

result_convolution = laplacian_sharpen(
    image,
    laplacian_4,
    "convolution"
)


# ============================================================
# COMPARE RESULTS
# ============================================================

difference = cv2.absdiff(
    result_correlation,
    result_convolution
)

max_difference = np.max(
    difference
)

mean_difference = np.mean(
    difference
)

identical = np.array_equal(
    result_correlation,
    result_convolution
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_dir = "../output/reflection2"

os.makedirs(
    output_dir,
    exist_ok=True
)

cv2.imwrite(
    os.path.join(
        output_dir,
        "correlation_result.png"
    ),
    result_correlation
)

cv2.imwrite(
    os.path.join(
        output_dir,
        "convolution_result.png"
    ),
    result_convolution
)

cv2.imwrite(
    os.path.join(
        output_dir,
        "difference.png"
    ),
    difference
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("REFLECTION QUESTION 2")
print("CORRELATION vs TRUE CONVOLUTION")
print("=" * 70)

print()
print("4-neighbor Laplacian kernel:")
print(laplacian_4)

print()
print("Maximum pixel difference :", max_difference)
print("Mean pixel difference    :", mean_difference)
print("Outputs identical        :", identical)

print()
print("Results saved to:")
print(output_dir)

print("=" * 70)