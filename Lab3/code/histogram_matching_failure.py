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
# IMAGE PATHS
# ============================================================

source_path = os.path.join(
    dataset_folder,
    "source1.jpg"
)

reference_path = os.path.join(
    dataset_folder,
    "chest_xray.png"
)


# ============================================================
# READ IMAGES
# ============================================================

source = cv2.imread(
    source_path,
    cv2.IMREAD_GRAYSCALE
)

reference = cv2.imread(
    reference_path,
    cv2.IMREAD_GRAYSCALE
)


if source is None:
    print("Source image not found!")
    exit()


if reference is None:
    print("Reference image not found!")
    exit()


# ============================================================
# HISTOGRAM FUNCTION
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
# CDF FUNCTION
# ============================================================

def calculate_cdf(histogram):

    cdf = np.cumsum(histogram)

    cdf = cdf / cdf[-1]

    return cdf


# ============================================================
# HISTOGRAM MATCHING
# ============================================================

source_histogram = calculate_histogram(
    source
)

reference_histogram = calculate_histogram(
    reference
)


source_cdf = calculate_cdf(
    source_histogram
)

reference_cdf = calculate_cdf(
    reference_histogram
)


# ============================================================
# CREATE LOOKUP TABLE
# ============================================================

lookup_table = np.zeros(
    256,
    dtype=np.uint8
)


for intensity in range(256):

    difference = np.abs(
        reference_cdf -
        source_cdf[intensity]
    )

    mapped_intensity = np.argmin(
        difference
    )

    lookup_table[intensity] = (
        mapped_intensity
    )


# ============================================================
# APPLY MATCHING
# ============================================================

matched = lookup_table[
    source
]


# ============================================================
# SAVE OUTPUT
# ============================================================

output_path = os.path.join(
    output_folder,
    "failure_case_matched.png"
)

cv2.imwrite(
    output_path,
    matched
)


# ============================================================
# DISPLAY COMPARISON
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
    "Source Image"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    reference,
    cmap="gray"
)

plt.title(
    "Very Different Reference"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    matched,
    cmap="gray"
)

plt.title(
    "Histogram Matched Result"
)

plt.axis("off")


plt.tight_layout()


plt.savefig(
    os.path.join(
        output_folder,
        "failure_case_comparison.png"
    )
)

plt.show()


# ============================================================
# HISTOGRAM COMPARISON
# ============================================================

matched_histogram = calculate_histogram(
    matched
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    source_histogram,
    label="Source"
)

plt.plot(
    reference_histogram,
    label="Reference"
)

plt.plot(
    matched_histogram,
    label="Matched Result"
)


plt.title(
    "Failure Case - Histogram Comparison"
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
        "failure_case_histograms.png"
    )
)

plt.show()


# ============================================================
# INFORMATION
# ============================================================

print("\n======================================")
print(" HISTOGRAM MATCHING FAILURE CASE")
print("======================================")

print(
    "\nSource image and reference image "
    "have very different content."
)

print(
    "\nHistogram matching considers only "
    "pixel intensity distributions."
)

print(
    "\nIt does not understand image content "
    "or semantic structures."
)

print(
    "\nTherefore, forcing the source to adopt "
    "the reference distribution can produce "
    "unnatural tonal changes and artifacts."
)

print(
    "\nOutput saved:"
)

print(
    "failure_case_matched.png"
)

print(
    "failure_case_comparison.png"
)

print(
    "failure_case_histograms.png"
)