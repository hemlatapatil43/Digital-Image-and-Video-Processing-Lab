DIGITAL IMAGE AND VIDEO PROCESSING LAB
LAB 3 - SPATIAL FILTERING

Title:
Restoring and Enhancing Degraded Onboard Camera Feed for an
Autonomous Ground Vehicle (AGV)


============================================================
1. AIM
============================================================

To implement and evaluate spatial filtering techniques for
restoring and enhancing a degraded AGV camera image.

The experiment includes:

1. Averaging filter for noise suppression.
2. Laplacian sharpening for boundary enhancement.
3. Unsharp masking and high-boost filtering.
4. Quantitative evaluation using PSNR and sharpness.
5. Analysis of the effect of noise level and filter parameters.
6. Recommendation of a suitable denoise-to-sharpen pipeline.


============================================================
2. DATASET
============================================================

A grayscale outdoor AGV-like image was used as the clean
ground-truth image.

The following degraded images were generated programmatically:

1. Gaussian noise with sigma = 10
2. Gaussian noise with sigma = 25
3. Motion blur using a linear motion-blur kernel

Dataset files:

dataset/
    agv_clean.jpg
    agv_clean.png
    agv_noisy_sigma10.png
    agv_noisy_sigma25.png
    agv_motion_blur.png


============================================================
3. TASK 1 - NOISE SUPPRESSION USING AVERAGING FILTER
============================================================

A 2D averaging/box filter was implemented from scratch using
a custom correlation operation.

No cv2.filter2D or scipy.signal filtering shortcut was used.

Kernel sizes tested:

    3 x 3
    5 x 5
    9 x 9

Noise levels tested:

    sigma = 10
    sigma = 25


RESULTS:

Noise       Kernel      PSNR (dB)      Sharpness
------------------------------------------------
Sigma 10    3x3         22.83          287.85
Sigma 10    5x5         21.06           67.17
Sigma 10    9x9         19.64           13.73

Sigma 25    3x3         22.14          479.76
Sigma 25    5x5         20.86          104.14
Sigma 25    9x9         19.59           19.63


OBSERVATION:

The 3x3 averaging filter provided the best PSNR among the
tested averaging filters.

Increasing the kernel size increased smoothing but caused
progressively greater loss of edges and fine image details.

The 9x9 filter produced strong smoothing and very low
sharpness, indicating that important high-frequency
information was removed.


============================================================
4. TASK 2 - LAPLACIAN SHARPENING
============================================================

Laplacian sharpening was applied to the motion-blurred image.

Two Laplacian kernels were implemented:

4-neighbor kernel:

    0  -1   0
   -1   4  -1
    0  -1   0


8-neighbor kernel:

   -1  -1  -1
   -1   8  -1
   -1  -1  -1


RESULTS:

Method                   PSNR (dB)      Sharpness
--------------------------------------------------
Motion blurred           21.05          372.42
4-neighbor Laplacian     18.62         6681.63
8-neighbor Laplacian     13.77        26924.67


OBSERVATION:

The 4-neighbor Laplacian provided a more controlled sharpening
result than the 8-neighbor version.

The 8-neighbor kernel responds more strongly to diagonal as well
as horizontal and vertical changes. This produced a very high
sharpness value but also amplified unwanted high-frequency
components.

Therefore, the 4-neighbor Laplacian is more suitable for this
AGV preprocessing experiment.


============================================================
5. TASK 3 - UNSHARP MASKING AND HIGH-BOOST FILTERING
============================================================

Unsharp masking and high-boost filtering were evaluated using
the sigma = 10 noisy image.

The following boost factors were tested:

    k = 1
    k = 1.5
    k = 2
    k = 3


RESULTS:

k         PSNR (dB)       Sharpness
-----------------------------------
1         21.26           24597.96
1.5       18.99           35485.69
2         17.30           47076.00
3         14.96           70570.92


OBSERVATION:

Increasing k increased the measured sharpness significantly.

However, PSNR continuously decreased.

This indicates that a larger boost factor does not necessarily
produce better image quality. At high values of k, noise and
unwanted high-frequency components are strongly amplified.

Therefore, k = 1 provides the best balance among the tested
values.


============================================================
6. TASK 4 - OBJECTIVE EVALUATION
============================================================

Two objective metrics were used.

PSNR:
Peak Signal-to-Noise Ratio was calculated against the clean
ground-truth image.

Sharpness:
Variance of the Laplacian was used as a simple measure of
high-frequency detail and edge strength.

Complete evaluation:

Configuration                         PSNR (dB)      Sharpness
----------------------------------------------------------------
Task 1 - Sigma 10 - 3x3                 22.83          277.90
Task 1 - Sigma 10 - 5x5                 21.06           67.51
Task 1 - Sigma 10 - 9x9                 19.64           18.35
Task 1 - Sigma 25 - 3x3                 22.14          469.94
Task 1 - Sigma 25 - 5x5                 20.86          104.53
Task 1 - Sigma 25 - 9x9                 19.59           24.14

Task 2 - Motion Blur                    21.05          405.67
Task 2 - 4 Neighbor                     18.62         6819.68
Task 2 - 8 Neighbor                     13.77        27111.75

Task 3 - k=1                            21.26        24653.42
Task 3 - k=1.5                          18.99        35550.70
Task 3 - k=2                            17.30        47149.26
Task 3 - k=3                            14.96        70656.93


INTERPRETATION:

The highest PSNR among the tested Task 1 configurations was
obtained using the 3x3 averaging filter for sigma = 10.

Large averaging kernels suppress noise but also remove
important image details.

For sharpening, increasing k increases the sharpness metric,
but the corresponding PSNR decreases. Therefore, extremely
large sharpness values should not be interpreted as improved
image quality.


============================================================
7. TASK 5 - RECOMMENDED AGV PIPELINE
============================================================

Recommended pipeline:

    Degraded AGV image
            |
            v
    3x3 Averaging Filter
            |
            v
    Controlled Sharpening
            |
            v
    Enhanced AGV Image


The experiments indicate that excessive smoothing destroys
navigation-relevant details, while excessive sharpening
amplifies noise.

Therefore, a small averaging kernel followed by controlled
sharpening is preferable to aggressive filtering.

For this experiment, a 3x3 averaging filter is recommended
for initial noise suppression.

A conservative sharpening parameter should then be used rather
than a large high-boost factor.


============================================================
8. REFLECTION QUESTION 1
============================================================

Sharpening and denoising were performed in both possible orders.

Results:

Pipeline                  PSNR (dB)      Sharpness
---------------------------------------------------
Denoise -> Sharpen         22.82          821.57
Sharpen -> Denoise         22.77          750.83


CONCLUSION:

Denoising before sharpening produced slightly better PSNR and
higher sharpness.

Sharpening first causes the noise and unwanted high-frequency
components to be enhanced before the denoising stage.

Therefore, for this experiment:

    Denoise -> Sharpen

is preferred.


============================================================
9. REFLECTION QUESTION 2
============================================================

The correlation implementation was compared with true
convolution using the 4-neighbor Laplacian kernel.

Results:

Maximum pixel difference = 0
Mean pixel difference    = 0
Outputs identical       = True


CONCLUSION:

The outputs are identical because the 4-neighbor Laplacian
kernel is symmetric.

Flipping the kernel during convolution therefore produces the
same kernel.

Consequently, correlation and convolution produce identical
results for this particular kernel.


============================================================
10. REFLECTION QUESTION 3
============================================================

PSNR and human perception do not always agree.

A large averaging kernel can reduce pixel-level error and
suppress noise, but it can also remove edges and fine details.

For example, increasing the averaging kernel from 3x3 to 9x9
strongly reduces the measured sharpness.

This can make the image appear smoother while simultaneously
removing navigation-relevant obstacle boundaries.

PSNR measures the numerical similarity between the processed
image and the clean reference image. It does not understand
which edges or structures are important for AGV navigation.

Therefore, PSNR should be combined with an edge or sharpness
metric and visual/task-specific analysis.


============================================================
11. REFLECTION QUESTION 4
============================================================

The noise level was increased beyond sigma = 25 to investigate
the limits of spatial-domain preprocessing.

Results:

Sigma    PSNR Denoised    Sharpness    PSNR Sharpened    Sharpness
-------------------------------------------------------------------
10       22.88            270.15       22.81             820.55
25       22.18            464.85       21.53            1607.18
40       21.17            788.74       19.94            2914.85
60       19.75           1345.23       18.04            5166.08
80       18.49           1937.48       16.57            7549.89


CONCLUSION:

As the noise level increases, PSNR decreases progressively.

Although the sharpness metric increases after sharpening, this
does not mean that useful edges are being recovered. At high
noise levels, sharpening also amplifies noise.

The results indicate substantial degradation from approximately
sigma = 40 to sigma = 60 onward, with sigma = 80 producing a
particularly poor result.

This demonstrates a fundamental limitation of simple spatial
filtering. Once image information has been heavily corrupted,
sharpening cannot reconstruct the lost information.

For severe degradation, more advanced methods such as temporal
filtering, motion compensation, advanced denoising, or learned
image-restoration methods would be required.


============================================================
12. CONCLUSION
============================================================

Spatial filtering techniques were successfully implemented and
evaluated for an AGV camera preprocessing problem.

Averaging filters were effective for reducing Gaussian noise,
but increasing the kernel size caused loss of image details.

Laplacian sharpening enhanced high-frequency information and
object boundaries, but aggressive sharpening also amplified
unwanted components.

Unsharp and high-boost filtering demonstrated that increasing
the boost factor increases measured sharpness while reducing
PSNR.

The experiments also demonstrated that PSNR alone is not enough
for selecting an image-processing configuration because
navigation-relevant edge information must also be preserved.

The overall experiment supports a controlled denoise-then-
sharpen strategy rather than aggressive filtering.


============================================================
13. FILE STRUCTURE
============================================================

Lab3/

    README.txt

    code/
        task1_averaging_filter.py
        task2_laplacian.py
        task3_unsharp_highboost.py
        task4_evaluation.py
        reflection1_order.py
        reflection2_convolution.py
        reflection4_noise_limit.py

    dataset/
        agv_clean.jpg
        agv_clean.png
        agv_noisy_sigma10.png
        agv_noisy_sigma25.png
        agv_motion_blur.png

    output/
        task1/
        task2/
        task3/
        task4/
        reflection1/
        reflection2/
        reflection4/


============================================================
END OF LAB 3
============================================================