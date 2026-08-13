import cv2
import matplotlib.pyplot as plt
import os

# -----------------------------------------
# PATHS
# -----------------------------------------

# Path of input image
image_path = "../dataset/pexels.jpg"

# Output folder
output_folder = "../output"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


# -----------------------------------------
# READ IMAGE
# -----------------------------------------

image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# -----------------------------------------
# FUNCTION 1: RESIZE IMAGE
# -----------------------------------------

def resize_image():

    print("\n--- RESIZE IMAGE ---")

    width = int(input("Enter new width: "))
    height = int(input("Enter new height: "))

    resized_image = cv2.resize(image_rgb, (width, height))

    # Save output
    output_path = os.path.join(output_folder, "resized_image.png")
    cv2.imwrite(
        output_path,
        cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
    )

    # Display output
    plt.imshow(resized_image)
    plt.title("Resized Image")
    plt.axis("off")
    plt.show()

    print("Saved:", output_path)


# -----------------------------------------
# FUNCTION 2: GRAYSCALE IMAGE
# -----------------------------------------

def grayscale_image():

    print("\n--- GRAYSCALE IMAGE ---")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save output
    output_path = os.path.join(output_folder, "grayscale_image.png")
    cv2.imwrite(output_path, gray)

    # Display output
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")
    plt.show()

    print("Saved:", output_path)


# -----------------------------------------
# FUNCTION 3: ROTATE IMAGE
# -----------------------------------------

def rotate_image():

    print("\n--- ROTATE IMAGE ---")

    rotated = cv2.rotate(
        image_rgb,
        cv2.ROTATE_90_CLOCKWISE
    )

    # Save output
    output_path = os.path.join(output_folder, "rotated_image.png")
    cv2.imwrite(
        output_path,
        cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)
    )

    # Display output
    plt.imshow(rotated)
    plt.title("Rotated Image")
    plt.axis("off")
    plt.show()

    print("Saved:", output_path)


# -----------------------------------------
# FUNCTION 4: BLUR IMAGE
# -----------------------------------------

def blur_image():

    print("\n--- BLUR IMAGE ---")

    blurred = cv2.GaussianBlur(
        image_rgb,
        (15, 15),
        0
    )

    # Save output
    output_path = os.path.join(output_folder, "blurred_image.png")
    cv2.imwrite(
        output_path,
        cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR)
    )

    # Display output
    plt.imshow(blurred)
    plt.title("Blurred Image")
    plt.axis("off")
    plt.show()

    print("Saved:", output_path)


# -----------------------------------------
# FUNCTION 5: EDGE DETECTION
# -----------------------------------------

def edge_detection():

    print("\n--- EDGE DETECTION ---")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    # Save output
    output_path = os.path.join(output_folder, "edge_detection.png")
    cv2.imwrite(output_path, edges)

    # Display output
    plt.imshow(edges, cmap="gray")
    plt.title("Edge Detection")
    plt.axis("off")
    plt.show()

    print("Saved:", output_path)


# -----------------------------------------
# MAIN MENU
# -----------------------------------------

while True:

    print("\n==============================")
    print(" DIGITAL IMAGE PROCESSING")
    print("==============================")

    print("1. Resize Image")
    print("2. Convert to Grayscale")
    print("3. Rotate Image")
    print("4. Blur Image")
    print("5. Edge Detection")
    print("0. Exit")

    choice = input("\nEnter your choice: ")

    if choice == "1":
        resize_image()

    elif choice == "2":
        grayscale_image()

    elif choice == "3":
        rotate_image()

    elif choice == "4":
        blur_image()

    elif choice == "5":
        edge_detection()

    elif choice == "0":
        print("\nProgram closed.")
        break

    else:
        print("\nInvalid choice! Please enter a valid number.")