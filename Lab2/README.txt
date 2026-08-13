# LAB 2 – INTENSITY TRANSFORMATION TECHNIQUES

## 1. Aim

To implement and study intensity transformation techniques for digital image enhancement using Python and OpenCV.

The following transformations are performed:

1. Gamma Correction
2. Log Transformation
3. Image Negative

## 2. Objective

The objective of this experiment is to understand how intensity transformation techniques modify the pixel intensity values of an image and improve its visual appearance.

## 3. Software Requirements

* Python 3.x
* OpenCV
* NumPy
* Matplotlib
* VS Code / PyCharm / Jupyter Notebook
* Windows/Linux/macOS

## 4. Input Dataset

Input image:

`pexels.jpg`

The input image is stored in:

`dataset/pexels.jpg`

The same input image is used for all three intensity transformation techniques.

## 5. Libraries Used

### OpenCV

OpenCV is used for reading images, grayscale conversion, applying transformations, and saving the processed images.

### NumPy

NumPy is used for numerical operations and creation of the gamma lookup table.

### Matplotlib

Matplotlib is used to display the original and transformed images.

### OS

The OS module is used to create and manage the output directory.

# 6. Theory

Intensity transformation is a fundamental image processing technique in which the intensity values of pixels are modified according to a mathematical transformation function.

The general transformation can be represented as:

s = T(r)

where:

* r = input pixel intensity
* s = output pixel intensity
* T = transformation function

Intensity transformations are useful for image enhancement and improving the visibility of important image details.

# 7. Gamma Correction

Gamma correction is a nonlinear intensity transformation used to control the brightness of an image.

The transformation is:

s = 255 × (r / 255)^(1/γ)

where:

* r = input pixel intensity
* s = output pixel intensity
* γ = gamma value

In this experiment:

γ = 2.2

Gamma correction is useful for adjusting image brightness and compensating for nonlinear characteristics of display and imaging systems.

## Algorithm – Gamma Correction

1. Read the input image.
2. Convert the image to grayscale.
3. Set the gamma value to 2.2.
4. Create a lookup table for all intensity values from 0 to 255.
5. Apply the gamma transformation using the lookup table.
6. Display the original and gamma-corrected images.
7. Save the gamma-corrected image.
8. Stop.

# 8. Log Transformation

Log transformation expands the range of low-intensity pixel values and compresses high-intensity values.

The transformation is:

s = c × log(1 + r)

where:

* r = input pixel intensity
* s = output pixel intensity
* c = scaling constant

Log transformation is useful for enhancing details in darker regions of an image.

## Algorithm – Log Transformation

1. Read the input image.
2. Convert the image to grayscale.
3. Convert the image into floating-point format.
4. Apply the logarithmic transformation.
5. Normalize the transformed image to the range 0–255.
6. Convert the result to 8-bit unsigned integer format.
7. Display the original and log-transformed images.
8. Save the log-transformed image.
9. Stop.

# 9. Image Negative

Image negative transformation reverses the intensity values of an image.

For an 8-bit grayscale image:

s = 255 - r

where:

* r = input pixel intensity
* s = output pixel intensity

For example:

Input intensity 0 becomes 255, while input intensity 255 becomes 0.

Image negative is useful for enhancing details in certain images and is commonly used in applications such as medical image processing and photographic processing.

## Algorithm – Image Negative

1. Read the input image.
2. Convert the image to grayscale.
3. Invert the pixel intensity values.
4. Display the original and negative images.
5. Save the negative image.
6. Stop.

# 10. Procedure

1. Place `pexels.jpg` inside the `dataset` folder.
2. Open the Lab2 folder in VS Code.
3. Open the `code` folder.
4. Run `gamma_correction.py`.
5. Observe and save the gamma-corrected image.
6. Run `log_transformation.py`.
7. Observe and save the log-transformed image.
8. Run `negative.py`.
9. Observe and save the negative image.
10. Verify all processed images in the `output` folder.

# 11. Output

The following output images are generated:

1. `gamma_corrected.png`
2. `log_transformed.png`
3. `negative.png`

The outputs are stored in the `output` folder.

# 12. Folder Structure

Lab2/

```
README.txt

code/
    gamma_correction.py
    log_transformation.py
    negative.py

dataset/
    pexels.jpg

output/
    gamma_corrected.png
    log_transformed.png
    negative.png
```

# 13. Result

The gamma correction, log transformation, and image negative operations were successfully implemented using Python and OpenCV.

# 14. Conclusion

The experiment demonstrated three important intensity transformation techniques used in digital image processing. Gamma correction was used to modify image brightness, log transformation enhanced details in darker regions, and negative transformation inverted the intensity values of the image. The processed images were successfully displayed and saved for further analysis.
