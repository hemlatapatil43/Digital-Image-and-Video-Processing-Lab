import cv2
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
# NEGATIVE TRANSFORMATION
# -----------------------------------------

negative = cv2.bitwise_not(img)


# -----------------------------------------
# SAVE OUTPUT
# -----------------------------------------

output_path = os.path.join(
    output_folder,
    "negative.png"
)

cv2.imwrite(output_path, negative)

print("Negative image saved at:")
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
plt.imshow(negative, cmap="gray")
plt.title("Negative Image")
plt.axis("off")

plt.tight_layout()
plt.show()