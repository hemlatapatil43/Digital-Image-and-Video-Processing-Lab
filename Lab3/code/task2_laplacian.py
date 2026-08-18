import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# TASK 2 - LAPLACIAN SHARPENING
# ============================================================

output_dir = "../output/task2"
os.makedirs(output_dir, exist_ok=True)


# ------------------------------------------------------------
# Read images
# ------------------------------------------------------------

clean = cv2.imread(
    "../dataset/agv_clean.png",
    cv2.IMREAD_GRAYSCALE
)

motion_blurred = cv2.imread(
    "../dataset/agv_motion_blur.png",
    cv2.IMREAD_GRAYSCALE
)

if clean is None or motion_blurred is None:
    print("Error: Required images not found.")
    exit()


# ============================================================
# FROM-SCRATCH 2D CORRELATION
# ============================================================

def correlate2d(image, kernel):

    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)

    image_height, image_width = image.shape

    kernel_height, kernel_width = kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded = np.pad(
        image,
        (
            (pad_height, pad_height),
            (pad_width, pad_width)
        ),
        mode="constant",
        constant_values=0
    )

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    for i in range(image_height):

        for j in range(image_width):

            region = padded[
                i:i + kernel_height,
                j:j + kernel_width
            ]

            output[i, j] = np.sum(
                region * kernel
            )

    return output


# ============================================================
# LAPLACIAN KERNELS
# ============================================================

laplacian_4 = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=np.float32)


laplacian_8 = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
], dtype=np.float32)


# ============================================================
# LAPLACIAN RESPONSE
# ============================================================

response_4 = correlate2d(
    motion_blurred,
    laplacian_4
)

response_8 = correlate2d(
    motion_blurred,
    laplacian_8
)


# ============================================================
# SHARPENING
# ============================================================

sharpened_4 = (
    motion_blurred.astype(np.float32)
    + response_4
)

sharpened_8 = (
    motion_blurred.astype(np.float32)
    + response_8
)


sharpened_4 = np.clip(
    sharpened_4,
    0,
    255
).astype(np.uint8)


sharpened_8 = np.clip(
    sharpened_8,
    0,
    255
).astype(np.uint8)


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(original, processed):

    original = original.astype(np.float32)
    processed = processed.astype(np.float32)

    mse = np.mean(
        (original - processed) ** 2
    )

    if mse == 0:
        return float("inf")

    return 10 * np.log10(
        (255 ** 2) / mse
    )


# ============================================================
# SHARPNESS
# ============================================================

def calculate_sharpness(image):

    laplacian = cv2.Laplacian(
        image,
        cv2.CV_64F
    )

    return laplacian.var()


# ============================================================
# METRICS
# ============================================================

psnr_blurred = calculate_psnr(
    clean,
    motion_blurred
)

psnr_4 = calculate_psnr(
    clean,
    sharpened_4
)

psnr_8 = calculate_psnr(
    clean,
    sharpened_8
)


sharp_blurred = calculate_sharpness(
    motion_blurred
)

sharp_4 = calculate_sharpness(
    sharpened_4
)

sharp_8 = calculate_sharpness(
    sharpened_8
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 65)
print("TASK 2 - LAPLACIAN SHARPENING RESULTS")
print("=" * 65)

print(
    f"{'Method':<25}"
    f"{'PSNR (dB)':<15}"
    f"{'Sharpness':<15}"
)

print("-" * 65)

print(
    f"{'Motion blurred':<25}"
    f"{psnr_blurred:<15.2f}"
    f"{sharp_blurred:<15.2f}"
)

print(
    f"{'4-neighbor Laplacian':<25}"
    f"{psnr_4:<15.2f}"
    f"{sharp_4:<15.2f}"
)

print(
    f"{'8-neighbor Laplacian':<25}"
    f"{psnr_8:<15.2f}"
    f"{sharp_8:<15.2f}"
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

cv2.imwrite(
    "../output/task2/laplacian_4_sharpened.png",
    sharpened_4
)

cv2.imwrite(
    "../output/task2/laplacian_8_sharpened.png",
    sharpened_8
)


# ============================================================
# VISUAL COMPARISON
# ============================================================

plt.figure(figsize=(15, 8))


plt.subplot(2, 3, 1)

plt.imshow(
    motion_blurred,
    cmap="gray"
)

plt.title("Motion Blurred")
plt.axis("off")


plt.subplot(2, 3, 2)

plt.imshow(
    response_4,
    cmap="gray"
)

plt.title("4-Neighbor Laplacian Response")
plt.axis("off")


plt.subplot(2, 3, 3)

plt.imshow(
    response_8,
    cmap="gray"
)

plt.title("8-Neighbor Laplacian Response")
plt.axis("off")


plt.subplot(2, 3, 5)

plt.imshow(
    sharpened_4,
    cmap="gray"
)

plt.title("4-Neighbor Sharpened")
plt.axis("off")


plt.subplot(2, 3, 6)

plt.imshow(
    sharpened_8,
    cmap="gray"
)

plt.title("8-Neighbor Sharpened")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    "../output/task2/laplacian_comparison.png",
    dpi=200
)

plt.show()


print("\nTask 2 completed successfully.")