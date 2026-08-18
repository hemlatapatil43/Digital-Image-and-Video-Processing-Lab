import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# TASK 3 - UNSHARP MASKING / HIGH-BOOST FILTERING
# ============================================================

output_dir = "../output/task3"
os.makedirs(output_dir, exist_ok=True)


# ------------------------------------------------------------
# Read images
# ------------------------------------------------------------

clean = cv2.imread(
    "../dataset/agv_clean.png",
    cv2.IMREAD_GRAYSCALE
)

noisy = cv2.imread(
    "../dataset/agv_noisy_sigma10.png",
    cv2.IMREAD_GRAYSCALE
)

if clean is None or noisy is None:
    print("Error: Required images not found.")
    exit()


# ============================================================
# FROM-SCRATCH CORRELATION
# ============================================================

def correlate2d(image, kernel):

    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)

    h, w = image.shape
    kh, kw = kernel.shape

    ph = kh // 2
    pw = kw // 2

    padded = np.pad(
        image,
        (
            (ph, ph),
            (pw, pw)
        ),
        mode="constant",
        constant_values=0
    )

    output = np.zeros_like(
        image,
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
# GAUSSIAN-LIKE SMOOTHING KERNEL
# ============================================================

kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)

kernel = kernel / kernel.sum()


# ------------------------------------------------------------
# Blur image
# ------------------------------------------------------------

blurred = correlate2d(
    noisy,
    kernel
)


# ============================================================
# UNSHARP MASK
# ============================================================

high_frequency = (
    noisy.astype(np.float32)
    - blurred
)


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
# TEST DIFFERENT BOOST FACTORS
# ============================================================

k_values = [
    1,
    1.5,
    2,
    3
]

results = []


for k in k_values:

    print(
        f"Processing k = {k}"
    )

    sharpened = (
        noisy.astype(np.float32)
        + k * high_frequency
    )

    sharpened = np.clip(
        sharpened,
        0,
        255
    ).astype(np.uint8)

    # Save
    filename = f"k_{k}.png"

    cv2.imwrite(
        os.path.join(
            output_dir,
            filename
        ),
        sharpened
    )

    # Metrics
    psnr = calculate_psnr(
        clean,
        sharpened
    )

    sharpness = calculate_sharpness(
        sharpened
    )

    results.append(
        (
            k,
            psnr,
            sharpness
        )
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("TASK 3 RESULTS")
print("=" * 60)

print(
    f"{'k':<10}"
    f"{'PSNR (dB)':<20}"
    f"{'Sharpness':<20}"
)

print("-" * 60)

for k, psnr, sharpness in results:

    print(
        f"{k:<10}"
        f"{psnr:<20.2f}"
        f"{sharpness:<20.2f}"
    )


# ============================================================
# OUTPUT GRID
# ============================================================

fig, axes = plt.subplots(
    1,
    5,
    figsize=(18, 4)
)


axes[0].imshow(
    noisy,
    cmap="gray"
)

axes[0].set_title(
    "Noisy σ=10"
)

axes[0].axis("off")


for i, k in enumerate(k_values):

    sharpened = (
        noisy.astype(np.float32)
        + k * high_frequency
    )

    sharpened = np.clip(
        sharpened,
        0,
        255
    ).astype(np.uint8)

    axes[i + 1].imshow(
        sharpened,
        cmap="gray"
    )

    axes[i + 1].set_title(
        f"k = {k}"
    )

    axes[i + 1].axis("off")


plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "task3_highboost_grid.png"
    ),
    dpi=200
)

plt.show()


# ============================================================
# SHARPNESS vs k
# ============================================================

k_list = [
    result[0]
    for result in results
]

sharpness_list = [
    result[2]
    for result in results
]

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    k_list,
    sharpness_list,
    marker="o"
)

plt.xlabel(
    "Boost Factor k"
)

plt.ylabel(
    "Sharpness (Variance of Laplacian)"
)

plt.title(
    "Sharpness vs Boost Factor"
)

plt.grid(True)

plt.savefig(
    os.path.join(
        output_dir,
        "sharpness_vs_k.png"
    ),
    dpi=200
)

plt.show()


print("\nTask 3 completed successfully.")