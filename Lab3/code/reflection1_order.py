import cv2
import numpy as np
import os


# ============================================================
# FROM-SCRATCH 2D CORRELATION
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
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output


# ============================================================
# AVERAGING FILTER
# ============================================================

def averaging_filter(image, size=3):

    kernel = np.ones((size, size), dtype=np.float32)
    kernel = kernel / (size * size)

    result = correlate2d(image, kernel)

    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# UNSHARP MASKING
# ============================================================

def unsharp_mask(image, k=1):

    blurred = averaging_filter(image, 3)

    image_float = image.astype(np.float32)
    blurred_float = blurred.astype(np.float32)

    mask = image_float - blurred_float

    sharpened = image_float + k * mask

    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(original, processed):

    mse = np.mean(
        (original.astype(np.float32) -
         processed.astype(np.float32)) ** 2
    )

    if mse == 0:
        return float("inf")

    return 10 * np.log10((255 ** 2) / mse)


# ============================================================
# SHARPNESS
# Variance of Laplacian
# ============================================================

def calculate_sharpness(image):

    laplacian = cv2.Laplacian(
        image,
        cv2.CV_64F
    )

    return laplacian.var()


# ============================================================
# PATHS
# ============================================================

dataset_dir = "../dataset"
output_dir = "../output/reflection1"

os.makedirs(output_dir, exist_ok=True)

clean_path = os.path.join(
    dataset_dir,
    "agv_clean.png"
)

noisy_path = os.path.join(
    dataset_dir,
    "agv_noisy_sigma10.png"
)

clean = cv2.imread(
    clean_path,
    cv2.IMREAD_GRAYSCALE
)

noisy = cv2.imread(
    noisy_path,
    cv2.IMREAD_GRAYSCALE
)


# ============================================================
# PIPELINE A
# DENOISE → SHARPEN
# ============================================================

denoised_first = averaging_filter(
    noisy,
    3
)

pipeline_A = unsharp_mask(
    denoised_first,
    k=1
)


# ============================================================
# PIPELINE B
# SHARPEN → DENOISE
# ============================================================

sharpened_first = unsharp_mask(
    noisy,
    k=1
)

pipeline_B = averaging_filter(
    sharpened_first,
    3
)


# ============================================================
# METRICS
# ============================================================

psnr_A = calculate_psnr(
    clean,
    pipeline_A
)

sharpness_A = calculate_sharpness(
    pipeline_A
)

psnr_B = calculate_psnr(
    clean,
    pipeline_B
)

sharpness_B = calculate_sharpness(
    pipeline_B
)


# ============================================================
# SAVE RESULTS
# ============================================================

cv2.imwrite(
    os.path.join(
        output_dir,
        "denoise_then_sharpen.png"
    ),
    pipeline_A
)

cv2.imwrite(
    os.path.join(
        output_dir,
        "sharpen_then_denoise.png"
    ),
    pipeline_B
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 70)
print("REFLECTION QUESTION 1")
print("SHARPEN → DENOISE vs DENOISE → SHARPEN")
print("=" * 70)

print(
    f"{'Pipeline':<30}"
    f"{'PSNR (dB)':<15}"
    f"{'Sharpness':<15}"
)

print("-" * 70)

print(
    f"{'Denoise → Sharpen':<30}"
    f"{psnr_A:<15.2f}"
    f"{sharpness_A:<15.2f}"
)

print(
    f"{'Sharpen → Denoise':<30}"
    f"{psnr_B:<15.2f}"
    f"{sharpness_B:<15.2f}"
)

print("=" * 70)

print()
print("Results saved to:")
print(output_dir)