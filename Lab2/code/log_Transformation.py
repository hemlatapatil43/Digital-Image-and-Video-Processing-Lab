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
# CONVERT IMAGE TO FLOAT
# -----------------------------------------

img_float = img.astype(np.float32)


# -----------------------------------------
# APPLY LOG TRANSFORMATION
# -----------------------------------------

log_img = cv2.log(1 + img_float)


# -----------------------------------------
# NORMALIZE RESULT
# -----------------------------------------

log_img = cv2.normalize(
    log_img,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

log_img = np.uint8(log_img)


# -----------------------------------------
# SAVE OUTPUT
# -----------------------------------------

output_path = os.path.join(
    output_folder,
    "log_transformed.png"
)

cv2.imwrite(output_path, log_img)

print("Log transformed image saved at:")
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
plt.imshow(log_img, cmap="gray")
plt.title("Log Transformed")
plt.axis("off")

plt.tight_layout()
plt.show()