# LAB 1 – BASIC IMAGE PROCESSING OPERATIONS USING OPENCV

## 1. Aim

To perform basic image processing operations such as image resizing, grayscale conversion, image rotation, Gaussian blurring, and edge detection using Python and OpenCV.

## 2. Objective

The objective of this experiment is to understand and implement fundamental image processing operations on a digital image using Python libraries such as OpenCV and Matplotlib.

## 3. Software Requirements

* Python 3.x
* OpenCV
* Matplotlib
* Operating System: Windows/Linux/macOS
* VS Code / PyCharm / Jupyter Notebook

## 4. Input Dataset

Input image:

`pexels.jpg`

The input image is stored in:

`dataset/pexels.jpg`

## 5. Libraries Used

### OpenCV

OpenCV is used for image reading, color conversion, resizing, rotation, Gaussian filtering, and Canny edge detection.

### Matplotlib

Matplotlib is used to display the processed images.

### OS

The OS module is used to create and manage the output directory.

## 6. Image Processing Operations

### 6.1 Image Resizing

Image resizing changes the width and height of an image according to the dimensions provided by the user.

The OpenCV function used is:

`cv2.resize()`

### 6.2 Grayscale Conversion

A color image contains three channels: Red, Green, and Blue. Grayscale conversion converts the image into a single intensity channel.

The OpenCV function used is:

`cv2.cvtColor()`

with:

`cv2.COLOR_BGR2GRAY`

### 6.3 Image Rotation

Rotation changes the orientation of an image. In this experiment, the image is rotated by 90 degrees clockwise.

The OpenCV function used is:

`cv2.rotate()`

### 6.4 Gaussian Blurring

Gaussian blur is used to smooth an image and reduce small details and noise.

The OpenCV function used is:

`cv2.GaussianBlur()`

A `(15, 15)` Gaussian kernel is used.

### 6.5 Edge Detection

Edge detection identifies significant changes in intensity in an image. Canny edge detection is used in this experiment.

The OpenCV function used is:

`cv2.Canny()`

The threshold values used are:

* Lower threshold = 100
* Upper threshold = 200

## 7. Algorithm

1. Start the program.
2. Read the input image from the dataset folder.
3. Check whether the image was loaded successfully.
4. Convert the image from BGR to RGB for displaying using Matplotlib.
5. Display the image processing menu.
6. Select an operation from the menu.
7. Perform the selected operation.
8. Display the processed image.
9. Save the processed image in the output folder.
10. Repeat the process until the user selects Exit.
11. Stop the program.

## 8. Procedure

1. Place the input image `pexels.jpg` inside the `dataset` folder.
2. Open the `Lab1` folder in VS Code or another Python IDE.
3. Open the `code` folder.
4. Run `lab1.py`.
5. Select an operation from the menu.
6. For resizing, enter the required width and height.
7. View the processed image.
8. The processed image is automatically saved in the `output` folder.
9. Repeat for all available operations.

## 9. Output

The program generates the following output images:

1. Resized Image
2. Grayscale Image
3. Rotated Image
4. Gaussian Blurred Image
5. Canny Edge Detected Image

All output images are stored in the `output` folder.

## 10. Folder Structure

Lab1/

```
code/
    lab1.py

dataset/
    pexels.jpg

output/
    resized_image.png
    grayscale_image.png
    rotated_image.png
    blurred_image.png
    edge_detection.png

README.txt
```

## 11. Conclusion

The basic image processing operations were successfully implemented using Python and OpenCV. The experiment helped in understanding how an image can be resized, converted to grayscale, rotated, blurred, and processed for edge detection. The processed results were displayed and saved for further analysis.
