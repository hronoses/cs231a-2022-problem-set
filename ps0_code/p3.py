# CS231A Homework 0, Problem 3
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
path = 'G:\My Drive\Handbooks\CV\courses\cs231a\problem set\ps0_code'

def part_a():
    # ===== Problem 3a =====
    # Read in the images, image1.jpg and image2.jpg, as color images.
    # Hint: use io.imread to read in the files

    img1, img2 = None, None
    # BEGIN YOUR CODE HERE
    img1 = io.imread(path + '\image1.jpg')
    img2 = io.imread(path + '\image2.jpg')
    print(img1.shape, img2.shape)
    # END YOUR CODE HERE
    return img1, img2

def normalize_img(img):
    return (img - img.min()) / img.max()

def part_b(img1, img2):
    # ===== Problem 3b =====
    # Convert the images to double precision and rescale them
    # to stretch from minimum value 0 to maximum value 1.

    # BEGIN YOUR CODE HERE
    img1 = normalize_img(img1.astype('double'))
    img2 = normalize_img(img2.astype('double'))
    # END YOUR CODE HERE
    return img1, img2
    
def part_c(img1, img2):
    # ===== Problem 3c =====
    # Add the images together and re-normalize them
    # to have minimum value 0 and maximum value 1.
    # Display this image.
    sumImage = None
    
    # BEGIN YOUR CODE HERE
    sumImage = normalize_img(img1 + img2)
    # END YOUR CODE HERE
    return sumImage

def part_d(img1, img2):
    # ===== Problem 3d =====
    # Create a new image such that the left half of
    # the image is the left half of image1 and the
    # right half of the image is the right half of image2.

    newImage1 = None

    # BEGIN YOUR CODE HERE
    newImage1 = np.hstack((img1[:,:int(img1.shape[1] / 2), :], img2[:, int(img2.shape[1] / 2):, :]))
    # END YOUR CODE HERE
    return newImage1

def part_e(img1, img2):    
    # ===== Problem 3e =====
    # Using a for loop, create a new image such that every odd
    # numbered row is the corresponding row from image1 and the
    # every even row is the corresponding row from image2.
    # Hint: Remember that indices start at 0 and not 1 in Python.

    newImage = None

    # BEGIN YOUR CODE HERE
    newImage = np.zeros(img1.shape, dtype=np.double)
    newImage[::2,:,:] = img1[::2,:,:]
    newImage[1:,:,:][::2,:,:] = img2[1:,:,:][::2,:,:]
    # END YOUR CODE HERE
    return newImage

def part_f(img1, img2):     
    # ===== Problem 3f =====
    # Accomplish the same task as part e without using a for-loop.
    # The functions reshape and tile may be helpful here.

    newImage3 = None

    # BEGIN YOUR CODE HERE

    # END YOUR CODE HERE
    return newImage3

def part_g(img):         
    # ===== Problem 3g =====
    # Convert the result from part f to a grayscale image.
    # Display the grayscale image with a title.
    # Hint: use np.dot and the standard formula for converting RGB to grey
    # greyscale = R*0.299 + G*0.587 + B*0.114

    # BEGIN YOUR CODE HERE
    conv = np.array([0.299, 0.587, 0.114])
    img = img @ conv
    print(img.shape)
    # END YOUR CODE HERE
    return img

if __name__ == '__main__':
    img1, img2 = part_a()
    img1, img2 = part_b(img1, img2)
    sumImage = part_c(img1, img2)
    newImage1 = part_d(img1, img2)
    newImage2 = part_e(img1, img2)
    newImage3 = part_f(img1, img2)
    img = part_g(newImage2)
 
    plt.imshow(img, cmap='gray')
    plt.show()
    # plt.imshow(newImage2)
    # plt.show()
    # plt.imshow(newImage3)
    # plt.show()
