import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# TASK 1 - NOISE SUPPRESSION USING AVERAGING FILTER
# ============================================================

# ------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------

output_dir = "../output/task1"
os.makedirs(output_dir, exist_ok=True)


# ------------------------------------------------------------
# Read images
# ------------------------------------------------------------

clean = cv2.imread(
    "../dataset/agv_clean.png",
    cv2.IMREAD_GRAYSCALE
)

noisy_sigma10 = cv2.imread(
    "../dataset/agv_noisy_sigma10.png",
    cv2.IMREAD_GRAYSCALE
)

noisy_sigma25 = cv2.imread(
    "../dataset/agv_noisy_sigma25.png",
    cv2.IMREAD_GRAYSCALE
)


if clean is None or noisy_sigma10 is None or noisy_sigma25 is None:
    print("Error: Dataset images not found.")
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

    # Zero padding
    padded = np.pad(
        image,
        (
            (pad_height, pad_height),
            (pad_width, pad_width)
        ),
        mode="constant",
        constant_values=0
    )

    output = np.zeros_like(image, dtype=np.float32)

    # Correlation
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
# AVERAGING FILTER
# ============================================================

def averaging_filter(image, kernel_size):

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.float32
    )

    kernel = kernel / (
        kernel_size * kernel_size
    )

    filtered = correlate2d(
        image,
        kernel
    )

    filtered = np.clip(
        filtered,
        0,
        255
    )

    return filtered.astype(np.uint8)


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
# SHARPNESS METRIC
# Variance of Laplacian
# ============================================================

def calculate_sharpness(image):

    laplacian = cv2.Laplacian(
        image,
        cv2.CV_64F
    )

    return laplacian.var()


# ============================================================
# PROCESS ALL CONFIGURATIONS
# ============================================================

noise_images = {
    "Sigma 10": noisy_sigma10,
    "Sigma 25": noisy_sigma25
}

kernel_sizes = [3, 5, 9]

results = []


for noise_name, noisy_image in noise_images.items():

    for kernel_size in kernel_sizes:

        print(
            f"Processing {noise_name}, "
            f"{kernel_size}x{kernel_size}"
        )

        filtered = averaging_filter(
            noisy_image,
            kernel_size
        )

        # Save output
        filename = (
            f"{noise_name.replace(' ', '_')}_"
            f"{kernel_size}x{kernel_size}.png"
        )

        output_path = os.path.join(
            output_dir,
            filename
        )

        cv2.imwrite(
            output_path,
            filtered
        )

        # Metrics
        psnr = calculate_psnr(
            clean,
            filtered
        )

        sharpness = calculate_sharpness(
            filtered
        )

        results.append(
            (
                noise_name,
                kernel_size,
                psnr,
                sharpness
            )
        )


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 65)
print("TASK 1 RESULTS")
print("=" * 65)

print(
    f"{'Noise':<12}"
    f"{'Kernel':<12}"
    f"{'PSNR (dB)':<15}"
    f"{'Sharpness':<15}"
)

print("-" * 65)

for noise, kernel, psnr, sharpness in results:

    print(
        f"{noise:<12}"
        f"{kernel}x{kernel:<10}"
        f"{psnr:<15.2f}"
        f"{sharpness:<15.2f}"
    )


# ============================================================
# OUTPUT GRID
# ============================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(16, 8)
)


# First row - sigma 10

axes[0, 0].imshow(
    noisy_sigma10,
    cmap="gray"
)

axes[0, 0].set_title(
    "Noisy σ=10"
)

axes[0, 0].axis("off")


for index, kernel_size in enumerate(kernel_sizes):

    filtered = averaging_filter(
        noisy_sigma10,
        kernel_size
    )

    axes[0, index + 1].imshow(
        filtered,
        cmap="gray"
    )

    axes[0, index + 1].set_title(
        f"{kernel_size}×{kernel_size}"
    )

    axes[0, index + 1].axis("off")


# Second row - sigma 25

axes[1, 0].imshow(
    noisy_sigma25,
    cmap="gray"
)

axes[1, 0].set_title(
    "Noisy σ=25"
)

axes[1, 0].axis("off")


for index, kernel_size in enumerate(kernel_sizes):

    filtered = averaging_filter(
        noisy_sigma25,
        kernel_size
    )

    axes[1, index + 1].imshow(
        filtered,
        cmap="gray"
    )

    axes[1, index + 1].set_title(
        f"{kernel_size}×{kernel_size}"
    )

    axes[1, index + 1].axis("off")


plt.tight_layout()

grid_path = os.path.join(
    output_dir,
    "task1_averaging_grid.png"
)

plt.savefig(
    grid_path,
    dpi=200
)

plt.show()


print("\nTask 1 completed successfully.")
print("Output folder:", output_dir)