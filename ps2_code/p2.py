import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from p1 import *

'''
COMPUTE_EPIPOLE computes the epipole in homogenous coordinates
given matching points in two images and the fundamental matrix
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the Fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the first image
'''
def compute_epipole(points1, points2, F):
    l = points2 @ F
    u, s, v = np.linalg.svd(l)
    min_s = np.argmin(s)
    # total least squares problem. Soltuion - right singular vector that correspond to the lowest singular value
    e = v[min_s]
    return e/e[-1]

    
'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images. Do not divide the homographies by their 2,2 entry.
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    width, height = im2.shape
    T = np.eye(3)
    T[:, -1] = [-width/2, -height/2, 1]
    e2_trans = T @ e2
    e2_trans /= e2_trans[-1]
    alpha = np.sign(e2_trans[0])
    e_length = np.sqrt(e2_trans @ e2_trans)
    R = np.diag([alpha * e2_trans[0] / e_length]*3)
    R[-1, -1] = 1
    R[0, 1] = alpha * e2_trans[1] / e_length
    R[1, 0] = -alpha * e2_trans[1] / e_length
    f = (R @ T @ e2)[0]
    G = np.eye(3)
    G[2, 0] = - 1 / f
    H2 = np.linalg.inv(T) @ G @ R @ T
    # compute H1
    e_x = np.zeros((3,3))
    e_x[0,1] = -e2[2]
    e_x[0,2] = e2[1]
    e_x[1,2] = -e2[0]
    e_x = e_x - e_x.T
    M = e_x @ F + np.outer(e2, np.ones((1,3)))
    p2 = points2 @ H2.T
    p2 = p2 / p2[:, -1, np.newaxis] 
    p1 = points2 @ M.T @ H2.T
    p1 /= p1[:, -1, np.newaxis]
    a = np.linalg.lstsq(p1, p2[:, 0])
    HA = np.eye(3) 
    HA[0] = a[0]
    H1 = HA @ H2 @ M
    return H1, H2

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    # plot_epipolar_lines_on_images(points1, points2, im1, im2, F)
    # plt.show()

    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print("H1:\n", H1)
    print('')
    print("H2:\n", H2)

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
