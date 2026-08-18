import cv2
import numpy as np
import os
import csv


# ============================================================
# TASK 4 - OBJECTIVE EVALUATION
# ============================================================

BASE_DIR = "../"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

TASK4_DIR = os.path.join(
    OUTPUT_DIR,
    "task4"
)

os.makedirs(
    TASK4_DIR,
    exist_ok=True
)


# ============================================================
# LOAD CLEAN GROUND TRUTH
# ============================================================

clean_path = os.path.join(
    DATASET_DIR,
    "agv_clean.png"
)

clean = cv2.imread(
    clean_path,
    cv2.IMREAD_GRAYSCALE
)

if clean is None:
    print("ERROR: agv_clean.png not found.")
    exit()


# ============================================================
# FROM-SCRATCH 2D CORRELATION
# ============================================================

def correlate2d(image, kernel):

    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)

    h, w = image.shape

    kh, kw = kernel.shape

    ph = kh // 2
    pw = kw // 2

    padded = np.pad(
        image,
        (
            (ph, ph),
            (pw, pw)
        ),
        mode="constant",
        constant_values=0
    )

    output = np.zeros(
        (h, w),
        dtype=np.float32
    )

    for i in range(h):

        for j in range(w):

            region = padded[
                i:i + kh,
                j:j + kw
            ]

            output[i, j] = np.sum(
                region * kernel
            )

    return output


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
        original,
        processed
):

    original = original.astype(
        np.float32
    )

    processed = processed.astype(
        np.float32
    )

    mse = np.mean(
        (original - processed) ** 2
    )

    if mse == 0:

        return float("inf")

    return 10 * np.log10(
        (255.0 ** 2) / mse
    )


# ============================================================
# SHARPNESS
#
# Variance of Laplacian
# Higher value = stronger edge/high-frequency response
# ============================================================

def calculate_sharpness(image):

    laplacian_kernel = np.array(
        [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ],
        dtype=np.float32
    )

    laplacian = correlate2d(
        image,
        laplacian_kernel
    )

    return np.var(
        laplacian
    )


# ============================================================
# EVALUATE IMAGE
# ============================================================

def evaluate(
        label,
        path
):

    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        print(
            "WARNING: Could not read:",
            path
        )

        return None

    # Match dimensions
    if image.shape != clean.shape:

        image = cv2.resize(
            image,
            (
                clean.shape[1],
                clean.shape[0]
            )
        )

    psnr = calculate_psnr(
        clean,
        image
    )

    sharpness = calculate_sharpness(
        image
    )

    return [
        label,
        psnr,
        sharpness
    ]


# ============================================================
# RESULTS LIST
# ============================================================

results = []


# ============================================================
# TASK 1
# ============================================================

task1_dir = os.path.join(
    OUTPUT_DIR,
    "task1"
)

task1_files = [

    (
        "Task 1 - Sigma 10 - 3x3",
        "Sigma_10_3x3.png"
    ),

    (
        "Task 1 - Sigma 10 - 5x5",
        "Sigma_10_5x5.png"
    ),

    (
        "Task 1 - Sigma 10 - 9x9",
        "Sigma_10_9x9.png"
    ),

    (
        "Task 1 - Sigma 25 - 3x3",
        "Sigma_25_3x3.png"
    ),

    (
        "Task 1 - Sigma 25 - 5x5",
        "Sigma_25_5x5.png"
    ),

    (
        "Task 1 - Sigma 25 - 9x9",
        "Sigma_25_9x9.png"
    )
]


for label, filename in task1_files:

    path = os.path.join(
        task1_dir,
        filename
    )

    result = evaluate(
        label,
        path
    )

    if result:

        results.append(
            result
        )


# ============================================================
# TASK 2
# ============================================================

task2_dir = os.path.join(
    OUTPUT_DIR,
    "task2"
)


# Motion blur baseline
motion_blur_path = os.path.join(
    DATASET_DIR,
    "agv_motion_blur.png"
)

result = evaluate(
    "Task 2 - Motion Blur",
    motion_blur_path
)

if result:

    results.append(
        result
    )


task2_files = [

    (
        "Task 2 - 4 Neighbor",
        "laplacian_4_sharpened.png"
    ),

    (
        "Task 2 - 8 Neighbor",
        "laplacian_8_sharpened.png"
    )
]


for label, filename in task2_files:

    path = os.path.join(
        task2_dir,
        filename
    )

    result = evaluate(
        label,
        path
    )

    if result:

        results.append(
            result
        )


# ============================================================
# TASK 3
# ============================================================

task3_dir = os.path.join(
    OUTPUT_DIR,
    "task3"
)

task3_files = [

    (
        "Task 3 - k=1",
        "k_1.png"
    ),

    (
        "Task 3 - k=1.5",
        "k_1.5.png"
    ),

    (
        "Task 3 - k=2",
        "k_2.png"
    ),

    (
        "Task 3 - k=3",
        "k_3.png"
    )
]


for label, filename in task3_files:

    path = os.path.join(
        task3_dir,
        filename
    )

    result = evaluate(
        label,
        path
    )

    if result:

        results.append(
            result
        )


# ============================================================
# PRINT COMPLETE TABLE
# ============================================================

print("\n")

print("=" * 78)

print(
    "TASK 4 - COMPLETE OBJECTIVE EVALUATION"
)

print("=" * 78)

print(
    f"{'Configuration':<40}"
    f"{'PSNR (dB)':<15}"
    f"{'Sharpness':<20}"
)

print("-" * 78)


for label, psnr, sharpness in results:

    print(
        f"{label:<40}"
        f"{psnr:<15.2f}"
        f"{sharpness:<20.2f}"
    )


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    TASK4_DIR,
    "task4_results.csv"
)

with open(
        csv_path,
        "w",
        newline=""
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow(
        [
            "Configuration",
            "PSNR_dB",
            "Sharpness"
        ]
    )

    for row in results:

        writer.writerow(
            row
        )


print("\n")

print(
    "Number of configurations evaluated:",
    len(results)
)

print(
    "\nResults saved to:"
)

print(
    csv_path
)

print(
    "\nTask 4 evaluation completed successfully."
)