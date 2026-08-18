============================================================
DIGITAL IMAGE AND VIDEO PROCESSING LAB
LAB 2 - HISTOGRAM MATCHING AND CLAHE
============================================================


1. AIM
------------------------------------------------------------

To study and implement histogram matching (histogram
specification) and Contrast Limited Adaptive Histogram
Equalization (CLAHE) for digital image enhancement.

The experiment consists of two problems:

Problem 1:
Histogram Matching / Histogram Specification

Problem 2:
Adaptive Histogram Equalization with Contrast Limiting
(CLAHE)


============================================================
PROBLEM 1 - HISTOGRAM MATCHING
============================================================


2. PROBLEM STATEMENT
------------------------------------------------------------

A film restoration studio has several frames of the same
scene with different tonal characteristics because of
changing daylight, film aging, and different camera rolls.

The objective is to transform differently exposed source
frames so that their intensity distributions match the
distribution of a selected reference frame.

The experiment also considers:

1. Why histogram equalization is unsuitable.
2. Histogram and CDF convergence.
3. An analytically defined stylized target histogram.
4. Failure cases of histogram matching.


============================================================
3. THEORY - HISTOGRAM MATCHING
============================================================

Histogram matching, also called histogram specification, is
an image enhancement technique in which the histogram of a
source image is transformed to match the histogram of a
reference image.

If r represents the input intensity and s represents the
output intensity, the transformation can be written as:

                    s = T(r)

The transformation is obtained using the cumulative
distribution functions (CDFs) of the source and reference
images.

The source CDF is calculated from the source histogram.

The reference CDF is calculated from the reference histogram.

For each source intensity, the reference intensity having
the closest CDF value is selected.

This produces a lookup table which is used to transform the
source image.


============================================================
4. HISTOGRAM EQUALIZATION VS HISTOGRAM MATCHING
============================================================

Histogram equalization attempts to produce an approximately
uniform intensity distribution.

It does not use a particular reference image.

Therefore, it violates the main requirement of the film
restoration application.

The requirement is:

"All frames should have the same tonal look as a selected
reference frame."

Histogram matching satisfies this requirement because the
reference image determines the desired intensity
distribution.

Therefore:

Histogram Equalization:
Source -> Approximately uniform distribution

Histogram Matching:
Source -> Reference distribution


============================================================
5. HISTOGRAM MATCHING ALGORITHM
============================================================

Step 1:
Read the source and reference images.

Step 2:
Convert the images to grayscale.

Step 3:
Calculate the histogram of the source image.

Step 4:
Calculate the histogram of the reference image.

Step 5:
Calculate the CDF of the source histogram.

Step 6:
Calculate the CDF of the reference histogram.

Step 7:
For every source intensity from 0 to 255, find the
reference intensity whose CDF is closest to the source CDF.

Step 8:
Create a lookup table using the obtained intensity mapping.

Step 9:
Apply the lookup table to every pixel of the source image.

Step 10:
Save and display the histogram-matched image.

Step 11:
Compare the source, reference, and matched histograms.

Step 12:
Compare the source, reference, and matched CDFs.


============================================================
6. DATASET - PROBLEM 1
============================================================

The following images are used:

reference.jpg
source1.jpg
source2.jpg

The source images represent differently exposed versions
of a scene, while reference.jpg represents the desired
tonal appearance.


============================================================
7. FIRST-PRINCIPLES IMPLEMENTATION
============================================================

The histogram is calculated manually.

For an 8-bit grayscale image there are 256 possible
intensity values.

A histogram array of size 256 is created.

For every pixel:

histogram[pixel] = histogram[pixel] + 1

The CDF is calculated using the cumulative sum:

CDF = cumulative sum of histogram

The CDF is normalized between 0 and 1.

The mapping is then obtained by comparing the source CDF
with the reference CDF.

No ready-made histogram matching function is used.


============================================================
8. ANALYTICALLY DEFINED TARGET HISTOGRAM
============================================================

Histogram matching does not require an actual reference
image.

An analytically defined target histogram can also be used.

For the stylized experiment, a Gaussian-like target
distribution is created:

                -(x-mu)^2
p(x) = exp( --------------- )
                   2*sigma^2

The target distribution is intentionally centered toward
lower intensity values to produce a moody, shadow-heavy
appearance.

Parameters used:

Mean (mu) = 65

Sigma = 25

The analytical histogram is normalized and converted into
a CDF.

The source CDF is then matched to the analytical target CDF.


============================================================
9. HISTOGRAM MATCHING FAILURE CASE
============================================================

Histogram matching does not understand image content.

It only considers the statistical distribution of pixel
intensities.

Therefore, matching a source image with a very different
reference image can produce unnatural results.

Possible problems include:

1. Unnatural brightness.
2. Excessive darkening.
3. Loss of local details.
4. Exaggerated shadows.
5. Exaggerated highlights.
6. Tonal artifacts.

The important limitation is:

A similar histogram does not guarantee similar visual
content.

Histogram matching is therefore most appropriate when the
source and reference images have compatible scene content
and the main difference is tonal appearance.


============================================================
PROBLEM 2 - CLAHE
============================================================


10. PROBLEM STATEMENT
------------------------------------------------------------

Chest X-ray images contain important fine structures in the
lungs that may occupy a narrow intensity range.

At the same time, the image may contain very bright bones
and very dark background regions.

Global histogram equalization uses one histogram for the
entire image and may fail to enhance subtle local details.

The objective is to use Adaptive Histogram Equalization
with Contrast Limiting (CLAHE) to improve local contrast
while reducing noise amplification.


============================================================
11. THEORY - GLOBAL HISTOGRAM EQUALIZATION
============================================================

Global histogram equalization uses the histogram of the
entire image to calculate one transformation function.

The same transformation is applied to every pixel.

This is unsuitable when an image contains different
regions with very different intensity characteristics.

For example:

Dark lung region
Bright bone region
Dark background

All regions contribute to the same global histogram.

A transformation suitable for one region may therefore be
unsuitable for another region.

Consequently, subtle lung structures may remain poorly
enhanced or become distorted.


============================================================
12. THEORY - CLAHE
============================================================

CLAHE stands for:

Contrast Limited Adaptive Histogram Equalization.

CLAHE divides an image into small rectangular regions
called tiles.

Histogram equalization is performed independently within
each tile.

This makes the transformation dependent on local image
statistics rather than one global histogram.


============================================================
13. CLAHE ALGORITHM
============================================================

Step 1:
Read the grayscale chest X-ray image.

Step 2:
Divide the image into rectangular tiles.

Step 3:
Calculate the histogram for each tile.

Step 4:
Apply histogram equalization locally.

Step 5:
Limit histogram bins using the clip limit.

Step 6:
Redistribute the clipped histogram excess.

Step 7:
Calculate the local transformation.

Step 8:
Use interpolation between neighboring tiles to produce
smooth transitions.

Step 9:
Combine the locally enhanced regions.

Step 10:
Generate the final CLAHE image.


============================================================
14. CONTRAST LIMITING
============================================================

A major problem with adaptive histogram equalization is
noise amplification.

In a homogeneous region, a small amount of noise may
occupy a narrow histogram bin.

Without contrast limiting, this bin can be strongly
amplified.

CLAHE limits the height of histogram bins using a clip
limit.

If a histogram bin exceeds the clip limit:

1. The excess pixels are removed from the bin.
2. The excess is redistributed among other bins.
3. The local transformation becomes less aggressive.

Therefore, contrast limiting reduces excessive noise
amplification in homogeneous regions.


============================================================
15. BILINEAR INTERPOLATION
============================================================

Independent processing of tiles can produce visible
boundaries or block-like seams.

CLAHE reduces this problem by interpolating the mappings
from neighboring tiles.

Bilinear interpolation combines the transformations of
neighboring tiles according to the pixel's position.

Therefore, the transition between adjacent tiles becomes
smooth.

This reduces blocky artifacts and visible seams.


============================================================
16. CLIP LIMIT PARAMETER
============================================================

The clip limit controls how strongly local contrast is
enhanced.

Low clip limit:

- Less contrast enhancement.
- Less noise amplification.
- May fail to reveal weak details.

Moderate clip limit:

- Good local contrast.
- Better detail visibility.
- Controlled noise.

Very high clip limit:

- Strong contrast enhancement.
- Increased noise amplification.
- Possible artifacts.


============================================================
17. TILE SIZE PARAMETER
============================================================

Tile size controls the spatial scale of local enhancement.

Very large tiles:

- Behavior approaches global histogram equalization.
- Less local adaptation.
- Fine local details may not be enhanced sufficiently.

Very small tiles:

- Strong local adaptation.
- Higher sensitivity to noise.
- Possible artificial texture.
- Higher computational cost.

Moderate tile size:

- Good balance between local enhancement and stability.


============================================================
18. PARAMETER SWEEP
============================================================

The experiment evaluates the following clip limits:

0.5
1.0
2.0
4.0
8.0

The following tile sizes are evaluated:

2 x 2
4 x 4
8 x 8
16 x 16
32 x 32

For every parameter combination, the following metrics
are calculated:

1. Entropy
2. Local contrast


============================================================
19. ENTROPY
============================================================

Entropy measures the amount of information or randomness
in an image.

It is calculated as:

H = -SUM(p(i) * log2(p(i)))

where p(i) is the probability of intensity i.

Higher entropy generally indicates a richer intensity
distribution.

However, maximum entropy does not always mean maximum
visual quality because excessive noise can also increase
entropy.


============================================================
20. LOCAL CONTRAST
============================================================

Local contrast measures the variation of intensity within
local neighborhoods.

In this experiment, local standard deviation is used as a
measure of local contrast.

Higher local contrast indicates stronger local intensity
variation.

This metric helps determine whether CLAHE reveals more
local detail than global equalization.


============================================================
21. DATASET - PROBLEM 2
============================================================

Input image:

chest_xray.jpg

The image is a grayscale chest X-ray used for studying
local contrast enhancement.


============================================================
22. OUTPUTS
============================================================

Problem 1 outputs:

source1_matched.png
source2_matched.png
histogram_matching_results.png
histogram_comparison.png
cdf_comparison.png
stylized_moody.png
stylized_histogram_result.png
stylized_cdf.png
failure_case_matched.png
failure_case_comparison.png
failure_case_histograms.png


Problem 2 outputs:

global_equalization.png
clahe_result.png
original_global_clahe.png
histogram_comparison.png

clip_0.5.png
clip_1.0.png
clip_2.0.png
clip_4.0.png
clip_8.0.png

clip_limit_entropy.png
clip_limit_contrast.png

tile_2x2.png
tile_4x4.png
tile_8x8.png
tile_16x16.png
tile_32x32.png

tile_size_entropy.png
tile_size_contrast.png


============================================================
23. SOFTWARE REQUIREMENTS
============================================================

Python 3.x

Libraries:

1. OpenCV
2. NumPy
3. Matplotlib

Install required libraries using:

pip install opencv-python numpy matplotlib


============================================================
24. FOLDER STRUCTURE
============================================================

Lab3/

    README.txt

    code/

        histogram_matching.py
        stylized_histogram.py
        histogram_matching_failure.py
        clahe_analysis.py

    dataset/

        reference.jpg
        source1.jpg
        source2.jpg
        chest_xray.png

    output/

        problem1/

            source1_matched.png
            source2_matched.png
            histogram_matching_results.png
            histogram_comparison.png
            cdf_comparison.png
            stylized_moody.png
            stylized_histogram_result.png
            stylized_cdf.png
            failure_case_matched.png
            failure_case_comparison.png
            failure_case_histograms.png

        problem2/

            global_equalization.png
            clahe_result.png
            original_global_clahe.png
            histogram_comparison.png
            clip_0.5.png
            clip_1.0.png
            clip_2.0.png
            clip_4.0.png
            clip_8.0.png
            clip_limit_entropy.png
            clip_limit_contrast.png
            tile_2x2.png
            tile_4x4.png
            tile_8x8.png
            tile_16x16.png
            tile_32x32.png
            tile_size_entropy.png
            tile_size_contrast.png


============================================================
25. RESULT
============================================================

Histogram matching successfully transformed differently
exposed source images toward the tonal distribution of the
selected reference image.

The CDF-based transformation successfully demonstrated
histogram specification from first principles.

An analytically defined target histogram was also used to
create a stylized shadow-heavy appearance.

The failure experiment demonstrated that histogram matching
can produce unnatural results when the source and reference
images have substantially different visual content.

CLAHE successfully provided local contrast enhancement for
the chest X-ray image.

The parameter sweep demonstrated the effects of clip limit
and tile size on entropy and local contrast.


============================================================
26. CONCLUSION
============================================================

Histogram matching and CLAHE are useful image enhancement
techniques but solve different problems.

Histogram matching is appropriate when the desired tonal
distribution is known and a reference image or target
distribution is available.

Unlike global histogram equalization, histogram matching
preserves the desired reference distribution instead of
forcing every image toward a uniform histogram.

CLAHE is appropriate when local contrast enhancement is
required. It analyzes individual image tiles and enhances
local structures.

The contrast limiting mechanism prevents excessive
amplification of noise, while interpolation between
neighboring tiles reduces block boundaries.

The experiments also demonstrate that parameter selection
is important. Excessive clip limits can amplify noise,
while very small tiles can create unstable local
enhancement. Very large tiles reduce the local adaptive
behavior.

Therefore, both histogram matching and CLAHE should be
selected according to the image characteristics and the
desired enhancement objective.


============================================================

============================================================