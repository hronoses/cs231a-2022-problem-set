# CS231A Homework 0, Problem 4
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def part_a():
    # ===== Problem 4a =====
    # Read in image1 as a grayscale image. Take the singular value
    # decomposition of the image.
    # Hint: use io.imread to read in the image file

    img1 = io.imread('image1.jpg', as_gray=True)
    u, s, v = np.linalg.svd(img1)
    return u, s, v

def part_b(u, s, v):
    # ===== Problem 4b =====
    # Save and display the best rank 1 approximation 
    # of the (grayscale) image1.
    rank1approx = s[0] * np.outer(u[:, 0], v[0]) 
    return rank1approx

def part_c(u, s, v):
    # ===== Problem 4c =====
    # Save and display the best rank 20 approximation
    # of the (grayscale) image1.

    rank20approx = None

    # BEGIN YOUR CODE HERE
    # rank20approx = np.sum([s[i] * np.outer(u[:, i], v[i])  for i in range(20)], axis=0)
    k = 20
    sm = np.eye(k)
    sm[np.diag_indices(k)] = s[:k]
    rank20approx = u[:, :k] @ sm @ v[:k, :]
    # END YOUR CODE HERE
    return rank20approx


if __name__ == '__main__':
    u, s, v = part_a()
    rank1approx = part_b(u, s, v)
    rank20approx = part_c(u, s, v)
    # plt.imshow(rank1approx)
    plt.imshow(rank20approx, cmap='gray')
    plt.show()