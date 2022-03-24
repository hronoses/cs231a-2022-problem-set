# CS231A Homework 1, Problem 2
import numpy as np

'''
DATA FORMAT

In this problem, we provide and load the data for you. Recall that in the original
problem statement, there exists a grid of black squares on a white background. We
know how these black squares are setup, and thus can determine the locations of
specific points on the grid (namely the corners). We also have images taken of the
grid at a front image (where Z = 0) and a back image (where Z = 150). The data we
load for you consists of three parts: real_XY, front_image, and back_image. For a
corner (0,0), we may see it at the (137, 44) pixel in the front image and the
(148, 22) pixel in the back image. Thus, one row of real_XY will contain the numpy
array [0, 0], corresponding to the real XY location (0, 0). The matching row in
front_image will contain [137, 44] and the matching row in back_image will contain
[148, 22]
'''

'''
COMPUTE_CAMERA_MATRIX
Arguments:
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    camera_matrix - The calibrated camera matrix (3x4 matrix)
'''
def compute_camera_matrix(real_XY, front_image, back_image):
    # TODO: Fill in this code
    # Hint: reshape your values such that you have PM=p,
    # and use np.linalg.lstsq or np.linalg.pinv to solve for M.
    # See https://apimirror.com/numpy~1.11/generated/numpy.linalg.pinv
    #
    # Our solution has shapes for M=(8,), P=(48,8), and p=(48,)
    # Alternatively, you can set things up such that M=(4,2), P=(24,4), and p=(24,2)
    # Lastly, reshape and add the (0,0,0,1) row to M to have it be (3,4)
    # BEGIN YOUR CODE HERE
    # print(real_XY.shape)
    p = np.vstack((real_XY, real_XY))
    front_image = np.hstack((front_image, np.zeros(12).reshape((12, 1)), np.ones(12).reshape((12, 1))))
    
    back_image = np.hstack((back_image, np.ones(12).reshape((12, 1)) * 150, np.ones(12).reshape((12, 1))))
    # back_image = front_image
    P = np.vstack((front_image, back_image))
    m, res, rank, s = np.linalg.lstsq(P, p)
    # m = np.hstack((m, np.ones(4).reshape((4, 1))))

    print(res, rank, s)
    print(np.sqrt(np.sum(res) / 24))
    return m
    # END YOUR CODE HERE

'''
RMS_ERROR
Arguments:
     camera_matrix - The camera matrix of the calibrated camera
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    rms_error - The root mean square error of reprojecting the points back
                into the images
'''
def rms_error(camera_matrix, real_XY, front_image, back_image):
    # BEGIN YOUR CODE HERE
    p = np.vstack((real_XY, real_XY))
    front_image = np.hstack((front_image, np.zeros(12).reshape((12, 1)), np.ones(12).reshape((12, 1))))
    back_image = np.hstack((back_image, np.ones(12).reshape((12, 1)) * 150, np.ones(12).reshape((12, 1))))
    # back_image = front_image

    P = np.vstack((front_image, back_image))
    p_pred = P @ camera_matrix
    rms = np.sqrt(np.sum((p - p_pred) ** 2) / 24)
    print(p - p_pred)
    print((p - p_pred) / p)
    return rms
    pass
    # END YOUR CODE HERE

if __name__ == '__main__':
    import os
    cwd = os.getcwd()
    print('*'*50)
    print(cwd)
    # Loading the example coordinates setup
    real_XY = np.load('real_XY.npy')
    front_image = np.load('front_image.npy')
    back_image = np.load('back_image.npy')

    camera_matrix = compute_camera_matrix(real_XY, front_image, back_image)

    print("Camera Matrix:\n", camera_matrix)
    print()
    print("RMS Error: ", rms_error(camera_matrix, real_XY, front_image, back_image))
