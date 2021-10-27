#!/usr/bin/env python3

'''
Implementation of mean, median, Gaussian and bilateral Gaussian filters
'''

__author__ = 'Senyao Hou'
__license__ = 'MIT'
__version__ = '0.0.1'

import numpy as np
from skimage import io
from matplotlib import pyplot as plt

# Global Variables
S = 1
SIGMA_S = 3
SIGMA_R = 75
NUM_RANDOM_MATRIX = 100
IMAGE_PATHS = ['B1.png', 'B2.png', 'C.png']

def calculate_mean(A: np.ndarray, **kwargs) -> float:
    weight = 1/(A.shape[0] * A.shape[1])
    weights = np.full(A.shape, weight)
    return (A * weights).sum()

def calculate_median(A: np.ndarray, **kwargs) -> float:

    # Flatten the matrix and sort it in ascending order
    sorted_matrix = np.sort(A.flatten(), axis=None)

    # If size is odd number, get the middle value
    if len(sorted_matrix) % 2 == 1:
        idx = (len(sorted_matrix) - 1)/2
        median = sorted_matrix[int(idx)]

    # If size is even number, get the mean of the middle two values
    else:
        idx = len(sorted_matrix)/2
        median = (sorted_matrix[int(idx)] + sorted_matrix[int(idx) - 1])/2

    return median

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:

    # Generate a discrete x axis with the length of the given size
    x = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)

    # Calculate Gaussian distribution
    gauss1d = np.exp(-0.5 * np.square(x) / np.square(sigma))

    # Calculate joined distribution
    gauss2d = np.outer(gauss1d, gauss1d)

    # Normalise the distribution
    kernel = gauss2d / np.sum(gauss2d)
    return kernel

def calculate_gaussian(A: np.ndarray, kernel: np.ndarray) -> float:
    return (A * kernel).sum()

def grayscale_gaussian_kernel(A: np.ndarray, s: int, sigma_r: float) -> float:
    kernel = np.zeros(A.shape, dtype='int64')
    center_pixel = A[s,s]
    for i in range(0,A.shape[0]):
        for j in range(0,A.shape[1]):
            kernel[i,j] = A[i,j] - center_pixel
    gaussian = np.exp(-0.5 * np.square(kernel) / np.square(sigma_r))
    return gaussian


def test_cal_mean_median(size: int):

    mean_deviations = []
    median_deviations = []

    # Generate a given number of random matrices
    for i in range(size):
        # Generate a random shape for a matrix, size between 2 and 100
        random_shape = (np.random.randint(low=2, high=100, size=1)[0], np.random.randint(low=2, high=100, size=1)[0])
        random_matrix = np.random.rand(*random_shape)
        print(f"{i+1} matrix shape: {random_shape}")

        # compare result against numpy native functions and calculate aboslute deviations
        mean_deviation = np.abs(calculate_mean(random_matrix) - random_matrix.mean())
        mean_deviations.append(mean_deviation)
        median_deviation = np.abs(calculate_median(random_matrix) - np.median(random_matrix))
        median_deviations.append(median_deviation)
        
    # Get maximum value of the deviations
    print(f"Maximum mean deviation: {max(mean_deviations)}")
    print(f"Maximum median deviation: {max(median_deviations)}")
        
def apply_filter(A: np.ndarray, s: int, filter_func, sigma_s: float = None, sigma_r: float = None) -> np.ndarray:
    # Calculate filter size for a given s
    size = 2 * s + 1

    # Ensure matrix size is bigger than filter size
    if A.shape[0] < size or A.shape[1] < size:
        print("Image size is too small to filter. Lower the s value or find a bigger image.")
        return None

    # Pad the matrix with values of the edges
    padded = np.pad(A, pad_width=s, mode='edge')

    # Create an empty matrix with the same shape
    filtered = np.zeros(A.shape, dtype='int64')
    
    # Check if it's using the Gaussian filter and get the kernel
    if sigma_s:
        kernel = gaussian_kernel(size, sigma_s)
    else:
        kernel = None

    # Loop through each pixel and apply the given filter
    for i in range(0,A.shape[0]):
        for j in range(0,A.shape[1]):

            patch = padded[i:i+size,j:j+size]

            # If use bilateral filter
            if sigma_r:
                grayscale_gaussian = grayscale_gaussian_kernel(patch, s, sigma_r)
                gaussian = gaussian_kernel(size, sigma_s)
                unnorm_kernel = gaussian * grayscale_gaussian
                kernel = unnorm_kernel/unnorm_kernel.sum()

            filtered[i,j] = filter_func(A=patch, kernel=kernel)
    
    return filtered

def mean_filter(A: np.ndarray, s: int) -> np.ndarray:
    return apply_filter(A, s, calculate_mean)

def median_filter(A: np.ndarray, s: int) -> np.ndarray:
    return apply_filter(A, s, calculate_median)

def gaussian_filter(A: np.ndarray, s: int, sigma_s: float) -> np.ndarray:
    return apply_filter(A, s, calculate_gaussian, sigma_s)

def bilateral_gaussian_filter(A: np.ndarray, s: int, sigma_s: float, sigma_r: float) -> np.ndarray:
    return apply_filter(A, s, calculate_gaussian, sigma_s, sigma_r)


if __name__ == "__main__":
    # Programmieraufgabe 1.1
    test_cal_mean_median(NUM_RANDOM_MATRIX)

    # Programmieraufgabe 1.2
    for path in IMAGE_PATHS:
        img = io.imread(path)

        filtered_img = mean_filter(img, S)
        plt.gray()
        plt.title(f"Mean filter with s={S} for {path}")
        plt.imshow(filtered_img)
        plt.show()

        gaussian_filtered_img = gaussian_filter(img, S, SIGMA_S)
        plt.gray()
        plt.title(f"Gaussian filter with s={S}, sigma_s={SIGMA_S} for {path}")
        plt.imshow(gaussian_filtered_img)
        plt.show()

    # Programmieraufgabe 1.3
    for path in IMAGE_PATHS:
        img = io.imread(path)
        filtered_img = median_filter(img, S)
        plt.title(f"Median filter with s={S} for {path}")
        plt.gray()
        plt.imshow(filtered_img)
        plt.show()

    # Programmieraufgabe 1.4
    for path in IMAGE_PATHS:
        img = io.imread(path)
        filtered_img = bilateral_gaussian_filter(img, S, SIGMA_S, SIGMA_R)
        plt.title(f"Bilateral Gaussian filter with s={S}, sigma_s={SIGMA_S}, sigma_r={SIGMA_R} for {path}")
        plt.gray()
        plt.imshow(filtered_img)
        plt.show()
