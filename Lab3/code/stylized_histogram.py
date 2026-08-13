import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# PATHS
# ============================================================

dataset_folder = "../dataset"
output_folder = "../output/problem1"

os.makedirs(output_folder, exist_ok=True)


# ============================================================
# INPUT IMAGE
# ============================================================

source_path = os.path.join(
    dataset_folder,
    "source1.jpg"
)

source = cv2.imread(
    source_path,
    cv2.IMREAD_GRAYSCALE
)

if source is None:
    print("Source image not found!")
    exit()


# ============================================================
# FUNCTION: CALCULATE HISTOGRAM
# ============================================================

def calculate_histogram(image):

    histogram = np.zeros(
        256,
        dtype=np.int64
    )

    for pixel in image.flatten():

        histogram[pixel] += 1

    return histogram


# ============================================================
# FUNCTION: CALCULATE CDF
# ============================================================

def calculate_cdf(histogram):

    cdf = np.cumsum(histogram)

    cdf = cdf / cdf[-1]

    return cdf


# ============================================================
# CREATE ANALYTICAL TARGET HISTOGRAM
# ============================================================

intensity = np.arange(256)

mean = 65
sigma = 25

target_histogram = np.exp(
    -0.5 *
    ((intensity - mean) / sigma) ** 2
)


# Normalize target histogram

target_histogram = (
    target_histogram /
    np.sum(target_histogram)
)


# ============================================================
# SOURCE HISTOGRAM
# ============================================================

source_histogram = calculate_histogram(
    source
)


# ============================================================
# SOURCE CDF
# ============================================================

source_cdf = calculate_cdf(
    source_histogram
)


# ============================================================
# TARGET CDF
# ============================================================

target_cdf = np.cumsum(
    target_histogram
)

target_cdf = (
    target_cdf /
    target_cdf[-1]
)


# ============================================================
# CREATE LOOKUP TABLE
# ============================================================

lookup_table = np.zeros(
    256,
    dtype=np.uint8
)


for source_intensity in range(256):

    difference = np.abs(
        target_cdf -
        source_cdf[source_intensity]
    )

    target_intensity = np.argmin(
        difference
    )

    lookup_table[
        source_intensity
    ] = target_intensity


# ============================================================
# APPLY TRANSFORMATION
# ============================================================

stylized_image = lookup_table[
    source
]


# ============================================================
# SAVE IMAGE
# ============================================================

output_path = os.path.join(
    output_folder,
    "stylized_moody.png"
)

cv2.imwrite(
    output_path,
    stylized_image
)


# ============================================================
# CALCULATE OUTPUT HISTOGRAM
# ============================================================

output_histogram = calculate_histogram(
    stylized_image
)

output_cdf = calculate_cdf(
    output_histogram
)


# ============================================================
# DISPLAY IMAGES
# ============================================================

plt.figure(
    figsize=(12, 4)
)


plt.subplot(1, 3, 1)

plt.imshow(
    source,
    cmap="gray"
)

plt.title(
    "Original Source"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    stylized_image,
    cmap="gray"
)

plt.title(
    "Stylized Moody Result"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.plot(
    target_histogram,
    label="Target"
)

plt.plot(
    output_histogram /
    np.sum(output_histogram),
    label="Output"
)

plt.title(
    "Target vs Output Histogram"
)

plt.xlabel(
    "Pixel Intensity"
)

plt.ylabel(
    "Probability"
)

plt.legend()

plt.grid()


plt.tight_layout()


plt.savefig(
    os.path.join(
        output_folder,
        "stylized_histogram_result.png"
    )
)


plt.show()


# ============================================================
# CDF COMPARISON
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    target_cdf,
    label="Target CDF"
)

plt.plot(
    output_cdf,
    label="Output CDF"
)

plt.title(
    "Stylized Target vs Output CDF"
)

plt.xlabel(
    "Pixel Intensity"
)

plt.ylabel(
    "Cumulative Probability"
)

plt.legend()

plt.grid()


plt.savefig(
    os.path.join(
        output_folder,
        "stylized_cdf.png"
    )
)


plt.show()


# ============================================================
# INFORMATION
# ============================================================

print("\n===================================")
print(" STYLIZED HISTOGRAM MATCHING")
print("===================================")

print("\nTarget distribution:")
print("Mean =", mean)
print("Sigma =", sigma)

print("\nOutput saved:")

print(
    "1. stylized_moody.png"
)

print(
    "2. stylized_histogram_result.png"
)

print(
    "3. stylized_cdf.png"
)