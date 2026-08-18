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

reference_path = os.path.join(dataset_folder, "reference.jpg")
source1_path = os.path.join(dataset_folder, "source1.jpg")
source2_path = os.path.join(dataset_folder, "source2.jpg")


# ============================================================
# READ IMAGES
# ============================================================

reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
source1 = cv2.imread(source1_path, cv2.IMREAD_GRAYSCALE)
source2 = cv2.imread(source2_path, cv2.IMREAD_GRAYSCALE)


# Check images

if reference is None:
    print("Reference image not found!")
    exit()

if source1 is None:
    print("Source 1 image not found!")
    exit()

if source2 is None:
    print("Source 2 image not found!")
    exit()


# ============================================================
# FUNCTION 1: CALCULATE HISTOGRAM
# ============================================================

def calculate_histogram(image):

    histogram = np.zeros(256, dtype=np.int64)

    for pixel in image.flatten():
        histogram[pixel] += 1

    return histogram


# ============================================================
# FUNCTION 2: CALCULATE CDF
# ============================================================

def calculate_cdf(histogram):

    cdf = np.cumsum(histogram)

    # Normalize CDF to range 0-1
    cdf = cdf / cdf[-1]

    return cdf


# ============================================================
# FUNCTION 3: HISTOGRAM MATCHING
# ============================================================

def histogram_matching(source, reference):

    # Calculate histograms
    source_hist = calculate_histogram(source)
    reference_hist = calculate_histogram(reference)

    # Calculate CDFs
    source_cdf = calculate_cdf(source_hist)
    reference_cdf = calculate_cdf(reference_hist)

    # Create lookup table
    lookup_table = np.zeros(256, dtype=np.uint8)

    # Find closest reference intensity
    for source_intensity in range(256):

        difference = np.abs(
            reference_cdf - source_cdf[source_intensity]
        )

        closest_intensity = np.argmin(difference)

        lookup_table[source_intensity] = closest_intensity

    # Apply lookup table
    matched_image = lookup_table[source]

    return matched_image


# ============================================================
# PERFORM HISTOGRAM MATCHING
# ============================================================

matched_source1 = histogram_matching(
    source1,
    reference
)

matched_source2 = histogram_matching(
    source2,
    reference
)


# ============================================================
# SAVE OUTPUT IMAGES
# ============================================================

cv2.imwrite(
    os.path.join(
        output_folder,
        "source1_matched.png"
    ),
    matched_source1
)

cv2.imwrite(
    os.path.join(
        output_folder,
        "source2_matched.png"
    ),
    matched_source2
)


# ============================================================
# CALCULATE HISTOGRAMS
# ============================================================

reference_hist = calculate_histogram(reference)

source1_hist = calculate_histogram(source1)

source2_hist = calculate_histogram(source2)

matched1_hist = calculate_histogram(matched_source1)

matched2_hist = calculate_histogram(matched_source2)


# ============================================================
# CALCULATE CDFS
# ============================================================

reference_cdf = calculate_cdf(reference_hist)

source1_cdf = calculate_cdf(source1_hist)

source2_cdf = calculate_cdf(source2_hist)

matched1_cdf = calculate_cdf(matched1_hist)

matched2_cdf = calculate_cdf(matched2_hist)


# ============================================================
# DISPLAY IMAGES
# ============================================================

plt.figure(figsize=(12, 8))


plt.subplot(2, 3, 1)
plt.imshow(source1, cmap="gray")
plt.title("Source 1")
plt.axis("off")


plt.subplot(2, 3, 2)
plt.imshow(source2, cmap="gray")
plt.title("Source 2")
plt.axis("off")


plt.subplot(2, 3, 3)
plt.imshow(reference, cmap="gray")
plt.title("Reference")
plt.axis("off")


plt.subplot(2, 3, 4)
plt.imshow(matched_source1, cmap="gray")
plt.title("Source 1 Matched")
plt.axis("off")


plt.subplot(2, 3, 5)
plt.imshow(matched_source2, cmap="gray")
plt.title("Source 2 Matched")
plt.axis("off")


plt.subplot(2, 3, 6)
plt.imshow(reference, cmap="gray")
plt.title("Target Reference")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "histogram_matching_results.png"
    )
)

plt.show()


# ============================================================
# HISTOGRAM COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    reference_hist,
    label="Reference"
)

plt.plot(
    source1_hist,
    label="Source 1"
)

plt.plot(
    source2_hist,
    label="Source 2"
)

plt.plot(
    matched1_hist,
    label="Source 1 Matched"
)

plt.plot(
    matched2_hist,
    label="Source 2 Matched"
)

plt.title("Histogram Comparison")

plt.xlabel("Pixel Intensity")

plt.ylabel("Frequency")

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
# CDF COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    reference_cdf,
    label="Reference CDF"
)

plt.plot(
    source1_cdf,
    label="Source 1 CDF"
)

plt.plot(
    source2_cdf,
    label="Source 2 CDF"
)

plt.plot(
    matched1_cdf,
    label="Source 1 Matched CDF"
)

plt.plot(
    matched2_cdf,
    label="Source 2 Matched CDF"
)

plt.title("CDF Comparison")

plt.xlabel("Pixel Intensity")

plt.ylabel("Cumulative Probability")

plt.legend()

plt.grid()

plt.savefig(
    os.path.join(
        output_folder,
        "cdf_comparison.png"
    )
)

plt.show()


# ============================================================
# INFORMATION
# ============================================================

print("\n===================================")
print(" HISTOGRAM MATCHING COMPLETED")
print("===================================")

print("\nOutput files saved in:")

print(output_folder)

print("\nGenerated files:")

print("1. source1_matched.png")

print("2. source2_matched.png")

print("3. histogram_matching_results.png")

print("4. histogram_comparison.png")

print("5. cdf_comparison.png")