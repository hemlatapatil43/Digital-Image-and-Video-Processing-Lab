import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------------------
# PATHS
# -----------------------------------------

image_path = "../dataset/pexels.jpg"
output_folder = "../output"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


# -----------------------------------------
# READ IMAGE
# -----------------------------------------

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found!")
    exit()


# -----------------------------------------
# GAMMA VALUE
# -----------------------------------------

gamma = 2.2


# -----------------------------------------
# CREATE LOOKUP TABLE
# -----------------------------------------

table = np.array(
    [((i / 255.0) ** (1 / gamma)) * 255
     for i in np.arange(256)]
).astype("uint8")


# -----------------------------------------
# APPLY GAMMA CORRECTION
# -----------------------------------------

gamma_img = cv2.LUT(img, table)


# -----------------------------------------
# SAVE OUTPUT
# -----------------------------------------

output_path = os.path.join(
    output_folder,
    "gamma_corrected.png"
)

cv2.imwrite(output_path, gamma_img)

print("Gamma corrected image saved at:")
print(output_path)


# -----------------------------------------
# DISPLAY IMAGES
# -----------------------------------------

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gamma_img, cmap="gray")
plt.title("Gamma Corrected")
plt.axis("off")

plt.tight_layout()
plt.show()