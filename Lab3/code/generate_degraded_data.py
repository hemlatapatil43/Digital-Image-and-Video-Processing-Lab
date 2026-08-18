import cv2
import numpy as np
import os


# ============================================================
# AGV DEGRADED DATA GENERATION
# ============================================================

# Paths
input_path = "../dataset/agv_clean.jpg"
output_dir = "../dataset"

# ------------------------------------------------------------
# Read clean image
# ------------------------------------------------------------

image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Clean image not found!")
    print("Expected:", input_path)
    exit()

print("Clean image loaded successfully.")
print("Image size:", image.shape)


# ------------------------------------------------------------
# Function: Add Gaussian Noise
# ------------------------------------------------------------

def add_gaussian_noise(image, sigma):

    noise = np.random.normal(
        0,
        sigma,
        image.shape
    )

    noisy = image.astype(np.float32) + noise

    noisy = np.clip(
        noisy,
        0,
        255
    )

    return noisy.astype(np.uint8)


# ------------------------------------------------------------
# Function: Motion Blur
# ------------------------------------------------------------

def motion_blur(image, kernel_size=9):

    kernel = np.zeros(
        (kernel_size, kernel_size),
        dtype=np.float32
    )

    # Horizontal motion
    kernel[kernel_size // 2, :] = 1.0 / kernel_size

    blurred = cv2.filter2D(
        image,
        -1,
        kernel
    )

    return blurred


# ------------------------------------------------------------
# Save clean image as PNG
# ------------------------------------------------------------

cv2.imwrite(
    os.path.join(output_dir, "agv_clean.png"),
    image
)


# ------------------------------------------------------------
# Gaussian Noise σ = 10
# ------------------------------------------------------------

noisy_sigma10 = add_gaussian_noise(
    image,
    10
)

cv2.imwrite(
    os.path.join(output_dir, "agv_noisy_sigma10.png"),
    noisy_sigma10
)


# ------------------------------------------------------------
# Gaussian Noise σ = 25
# ------------------------------------------------------------

noisy_sigma25 = add_gaussian_noise(
    image,
    25
)

cv2.imwrite(
    os.path.join(output_dir, "agv_noisy_sigma25.png"),
    noisy_sigma25
)


# ------------------------------------------------------------
# Motion Blur
# ------------------------------------------------------------

blurred = motion_blur(
    image,
    9
)

cv2.imwrite(
    os.path.join(output_dir, "agv_motion_blur.png"),
    blurred
)


# ------------------------------------------------------------
# Display information
# ------------------------------------------------------------

print("\nDegraded dataset generated successfully!")

print("\nGenerated files:")

print("1. agv_clean.png")
print("2. agv_noisy_sigma10.png")
print("3. agv_noisy_sigma25.png")
print("4. agv_motion_blur.png")

print("\nNoise levels:")
print("Gaussian σ = 10")
print("Gaussian σ = 25")

print("\nMotion blur:")
print("Linear kernel size = 9 pixels")