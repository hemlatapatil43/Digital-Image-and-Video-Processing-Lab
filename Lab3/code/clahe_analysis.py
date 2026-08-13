import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# PATHS
# ============================================================

dataset_folder = "../dataset"
output_folder = "../output/problem2"

os.makedirs(output_folder, exist_ok=True)

image_path = os.path.join(
    dataset_folder,
    "chest_xray.png"
)


# ============================================================
# READ IMAGE
# ============================================================

image = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    print("Chest X-ray image not found!")
    exit()


# ============================================================
# FUNCTION 1: ENTROPY
# ============================================================

def calculate_entropy(image):

    histogram = cv2.calcHist(
        [image],
        [0],
        None,
        [256],
        [0, 256]
    )

    histogram = histogram.flatten()

    probability = histogram / np.sum(histogram)

    probability = probability[
        probability > 0
    ]

    entropy = -np.sum(
        probability * np.log2(probability)
    )

    return entropy


# ============================================================
# FUNCTION 2: LOCAL CONTRAST
# ============================================================

def calculate_local_contrast(image):

    image_float = image.astype(
        np.float32
    )

    mean = cv2.GaussianBlur(
        image_float,
        (0, 0),
        3
    )

    squared_mean = cv2.GaussianBlur(
        image_float ** 2,
        (0, 0),
        3
    )

    variance = squared_mean - mean ** 2

    variance = np.maximum(
        variance,
        0
    )

    local_std = np.sqrt(
        variance
    )

    return np.mean(local_std)


# ============================================================
# ORIGINAL IMAGE METRICS
# ============================================================

original_entropy = calculate_entropy(
    image
)

original_contrast = calculate_local_contrast(
    image
)


# ============================================================
# GLOBAL HISTOGRAM EQUALIZATION
# ============================================================

global_equalized = cv2.equalizeHist(
    image
)


# ============================================================
# CLAHE
# ============================================================

clip_limit = 2.0
tile_size = 8

clahe = cv2.createCLAHE(
    clipLimit=clip_limit,
    tileGridSize=(tile_size, tile_size)
)

clahe_image = clahe.apply(
    image
)


# ============================================================
# SAVE MAIN RESULTS
# ============================================================

cv2.imwrite(
    os.path.join(
        output_folder,
        "global_equalization.png"
    ),
    global_equalized
)

cv2.imwrite(
    os.path.join(
        output_folder,
        "clahe_result.png"
    ),
    clahe_image
)


# ============================================================
# DISPLAY ORIGINAL, GLOBAL AND CLAHE
# ============================================================

plt.figure(
    figsize=(12, 4)
)

plt.subplot(1, 3, 1)

plt.imshow(
    image,
    cmap="gray"
)

plt.title(
    "Original X-ray"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    global_equalized,
    cmap="gray"
)

plt.title(
    "Global Equalization"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    clahe_image,
    cmap="gray"
)

plt.title(
    "CLAHE"
)

plt.axis("off")


plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "original_global_clahe.png"
    )
)

plt.show()


# ============================================================
# HISTOGRAM COMPARISON
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    image.ravel(),
    bins=256,
    range=(0, 256),
    alpha=0.5,
    label="Original"
)

plt.hist(
    global_equalized.ravel(),
    bins=256,
    range=(0, 256),
    alpha=0.5,
    label="Global Equalization"
)

plt.hist(
    clahe_image.ravel(),
    bins=256,
    range=(0, 256),
    alpha=0.5,
    label="CLAHE"
)

plt.title(
    "Histogram Comparison"
)

plt.xlabel(
    "Pixel Intensity"
)

plt.ylabel(
    "Frequency"
)

plt.legend()

plt.grid()

plt.savefig(
    os.path.join(
        output_folder,
        "histogram_comparison.png"
    )
)

plt.show()


# ============================================================
# CALCULATE MAIN METRICS
# ============================================================

global_entropy = calculate_entropy(
    global_equalized
)

clahe_entropy = calculate_entropy(
    clahe_image
)

global_contrast = calculate_local_contrast(
    global_equalized
)

clahe_contrast = calculate_local_contrast(
    clahe_image
)


print("\n======================================")
print(" MAIN CLAHE RESULTS")
print("======================================")

print(
    f"Original Entropy       : {original_entropy:.4f}"
)

print(
    f"Global Equalization    : {global_entropy:.4f}"
)

print(
    f"CLAHE Entropy          : {clahe_entropy:.4f}"
)

print()

print(
    f"Original Local Contrast: {original_contrast:.4f}"
)

print(
    f"Global Local Contrast  : {global_contrast:.4f}"
)

print(
    f"CLAHE Local Contrast   : {clahe_contrast:.4f}"
)


# ============================================================
# CLIP LIMIT SWEEP
# ============================================================

clip_limits = [
    0.5,
    1.0,
    2.0,
    4.0,
    8.0
]

clip_entropy = []
clip_contrast = []


for limit in clip_limits:

    clahe_temp = cv2.createCLAHE(
        clipLimit=limit,
        tileGridSize=(8, 8)
    )

    result = clahe_temp.apply(
        image
    )

    entropy = calculate_entropy(
        result
    )

    contrast = calculate_local_contrast(
        result
    )

    clip_entropy.append(
        entropy
    )

    clip_contrast.append(
        contrast
    )

    cv2.imwrite(
        os.path.join(
            output_folder,
            f"clip_{limit}.png"
        ),
        result
    )


# ============================================================
# CLIP LIMIT PLOT — ENTROPY
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    clip_limits,
    clip_entropy,
    marker="o"
)

plt.xlabel(
    "Clip Limit"
)

plt.ylabel(
    "Entropy"
)

plt.title(
    "Clip Limit vs Entropy"
)

plt.grid()

plt.savefig(
    os.path.join(
        output_folder,
        "clip_limit_entropy.png"
    )
)

plt.show()


# ============================================================
# CLIP LIMIT PLOT — LOCAL CONTRAST
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    clip_limits,
    clip_contrast,
    marker="o"
)

plt.xlabel(
    "Clip Limit"
)

plt.ylabel(
    "Local Contrast"
)

plt.title(
    "Clip Limit vs Local Contrast"
)

plt.grid()

plt.savefig(
    os.path.join(
        output_folder,
        "clip_limit_contrast.png"
    )
)

plt.show()


# ============================================================
# TILE SIZE SWEEP
# ============================================================

tile_sizes = [
    2,
    4,
    8,
    16,
    32
]

tile_entropy = []
tile_contrast = []


for size in tile_sizes:

    clahe_temp = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(size, size)
    )

    result = clahe_temp.apply(
        image
    )

    entropy = calculate_entropy(
        result
    )

    contrast = calculate_local_contrast(
        result
    )

    tile_entropy.append(
        entropy
    )

    tile_contrast.append(
        contrast
    )

    cv2.imwrite(
        os.path.join(
            output_folder,
            f"tile_{size}x{size}.png"
        ),
        result
    )


# ============================================================
# TILE SIZE PLOT — ENTROPY
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    tile_sizes,
    tile_entropy,
    marker="o"
)

plt.xlabel(
    "Tile Size"
)

plt.ylabel(
    "Entropy"
)

plt.title(
    "Tile Size vs Entropy"
)

plt.grid()

plt.savefig(
    os.path.join(
        output_folder,
        "tile_size_entropy.png"
    )
)

plt.show()


# ============================================================
# TILE SIZE PLOT — LOCAL CONTRAST
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    tile_sizes,
    tile_contrast,
    marker="o"
)

plt.xlabel(
    "Tile Size"
)

plt.ylabel(
    "Local Contrast"
)

plt.title(
    "Tile Size vs Local Contrast"
)

plt.grid()

plt.savefig(
    os.path.join(
        output_folder,
        "tile_size_contrast.png"
    )
)

plt.show()


# ============================================================
# PARAMETER SUMMARY
# ============================================================

print("\n======================================")
print(" CLIP LIMIT SWEEP")
print("======================================")

for i in range(len(clip_limits)):

    print(
        f"Clip Limit = {clip_limits[i]:4.1f} | "
        f"Entropy = {clip_entropy[i]:.4f} | "
        f"Local Contrast = {clip_contrast[i]:.4f}"
    )


print("\n======================================")
print(" TILE SIZE SWEEP")
print("======================================")

for i in range(len(tile_sizes)):

    print(
        f"Tile Size = {tile_sizes[i]:2d}x{tile_sizes[i]:2d} | "
        f"Entropy = {tile_entropy[i]:.4f} | "
        f"Local Contrast = {tile_contrast[i]:.4f}"
    )


print("\n======================================")
print(" CLAHE ANALYSIS COMPLETED")
print("======================================")