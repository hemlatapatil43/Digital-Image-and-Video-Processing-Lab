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

    output = np.zeros(
        (h, w),
        dtype=np.float32
    )

    for i in range(h):
        for j in range(w):

            region = padded[
                i:i + kh,
                j:j + kw
            ]

            output[i, j] = np.sum(
                region * kernel
            )

    return output


# ============================================================
# 3x3 AVERAGING FILTER
# ============================================================

def averaging_filter(image):

    kernel = np.ones(
        (3, 3),
        dtype=np.float32
    ) / 9.0

    result = correlate2d(
        image,
        kernel
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# UNSHARP MASKING k=1
# ============================================================

def unsharp_mask(image, k=1):

    blurred = averaging_filter(
        image
    )

    image_float = image.astype(
        np.float32
    )

    blurred_float = blurred.astype(
        np.float32
    )

    mask = image_float - blurred_float

    sharpened = (
        image_float +
        k * mask
    )

    return np.clip(
        sharpened,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# ADD GAUSSIAN NOISE
# ============================================================

def add_gaussian_noise(image, sigma):

    noise = np.random.normal(
        0,
        sigma,
        image.shape
    )

    noisy = (
        image.astype(np.float32)
        + noise
    )

    return np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
    original,
    processed
):

    mse = np.mean(
        (
            original.astype(np.float32)
            -
            processed.astype(np.float32)
        ) ** 2
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
# PATHS
# ============================================================

dataset_dir = "../dataset"

output_dir = (
    "../output/"
    "reflection4"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# ============================================================
# LOAD CLEAN IMAGE
# ============================================================

clean_path = os.path.join(
    dataset_dir,
    "agv_clean.png"
)

clean = cv2.imread(
    clean_path,
    cv2.IMREAD_GRAYSCALE
)

if clean is None:

    raise FileNotFoundError(
        "Could not load "
        "agv_clean.png"
    )


# ============================================================
# FIX RANDOM SEED
# Makes experiment reproducible
# ============================================================

np.random.seed(42)


# ============================================================
# NOISE LEVELS
# ============================================================

sigma_values = [
    10,
    25,
    40,
    60,
    80
]


results = []


# ============================================================
# PROCESS EACH NOISE LEVEL
# ============================================================

for sigma in sigma_values:

    print(
        f"Processing sigma = {sigma}"
    )

    # --------------------------------------------------------
    # Generate noisy image
    # --------------------------------------------------------

    noisy = add_gaussian_noise(
        clean,
        sigma
    )

    # --------------------------------------------------------
    # Denoise
    # --------------------------------------------------------

    denoised = averaging_filter(
        noisy
    )

    # --------------------------------------------------------
    # Sharpen
    # --------------------------------------------------------

    sharpened = unsharp_mask(
        denoised,
        k=1
    )

    # --------------------------------------------------------
    # Metrics before sharpening
    # --------------------------------------------------------

    psnr_noisy = calculate_psnr(
        clean,
        denoised
    )

    sharpness_noisy = calculate_sharpness(
        denoised
    )

    # --------------------------------------------------------
    # Metrics after sharpening
    # --------------------------------------------------------

    psnr_sharpened = calculate_psnr(
        clean,
        sharpened
    )

    sharpness_sharpened = calculate_sharpness(
        sharpened
    )

    # --------------------------------------------------------
    # Save images
    # --------------------------------------------------------

    cv2.imwrite(
        os.path.join(
            output_dir,
            f"sigma_{sigma}_noisy.png"
        ),
        noisy
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            f"sigma_{sigma}_denoised.png"
        ),
        denoised
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            f"sigma_{sigma}_sharpened.png"
        ),
        sharpened
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append(
        (
            sigma,
            psnr_noisy,
            sharpness_noisy,
            psnr_sharpened,
            sharpness_sharpened
        )
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 95)

print(
    "REFLECTION QUESTION 4"
)

print(
    "NOISE LIMIT OF SPATIAL-DOMAIN PREPROCESSING"
)

print("=" * 95)

print(
    f"{'Sigma':<10}"
    f"{'PSNR Denoised':<18}"
    f"{'Sharp. Denoised':<18}"
    f"{'PSNR Sharpened':<18}"
    f"{'Sharp. Sharpened':<18}"
)

print("-" * 95)


for result in results:

    sigma = result[0]
    psnr_d = result[1]
    sharp_d = result[2]
    psnr_s = result[3]
    sharp_s = result[4]

    print(
        f"{sigma:<10}"
        f"{psnr_d:<18.2f}"
        f"{sharp_d:<18.2f}"
        f"{psnr_s:<18.2f}"
        f"{sharp_s:<18.2f}"
    )


print("=" * 95)

print()
print(
    "Results saved to:"
)

print(
    output_dir
)