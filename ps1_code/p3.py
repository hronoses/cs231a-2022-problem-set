# CS231A Homework 1, Problem 3
import numpy as np
from utils import mat2euler
import math

'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - a list of all the points where each row is (x, y). 
            It will contain four points: two for each parallel line.
Returns:
    vanishing_point - the pixel location of the vanishing point
'''
def compute_vanishing_point(points):
    # Simple solution based on normals
    vp = np.hstack((points, np.ones((4,1))))
    l1 = np.cross(vp[0], vp[1])
    l2 = np.cross(vp[2], vp[3])
    x = np.cross(l1, l2)
    x = x[:2] / x[-1]
    # Solution based on line directions
    # q1 = points[0] - points[1]
    # q2 = points[2] - points[3]
    # ori = points[0] - points[2]
    # t = np.linalg.det(np.vstack((ori, q2))) / np.linalg.det(np.vstack((q1, q2)))
    # x = points[0] - t * q1
    return x

'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''
def compute_K_from_vanishing_points(vanishing_points):
    A = np.zeros((3,3))
    p = np.array(vanishing_points)
    for i, (k, m) in enumerate([[0,1], [0,2], [1,2]]):
        A[i, 0] = p[k,0] * p[m,0] +  p[m,1] * p[k,1]
        A[i, 1] = p[k,0] + p[m,0]
        A[i, 2] = p[k,1] + p[m,1]
    x = np.linalg.solve(A, np.ones(3) * (-1))    
    w = np.zeros((3,3))
    w[0, 0] = x[0]
    w[0, 2] = x[1]
    w[2, 0] = x[1]
    w[1, 1] = x[0]
    w[1, 2] = x[2]
    w[2, 1] = x[2]
    w[2, 2] = 1
    K_inv = np.linalg.cholesky(w)
    K = np.linalg.inv(K_inv.T)
    # K /= K[-1,-1] 
    return K

'''
COMPUTE_ANGLE_BETWEEN_PLANES
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''
def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    vp1 = np.hstack((vanishing_pair1, np.ones((2,1))))
    vp2 = np.hstack((vanishing_pair2, np.ones((2,1))))
    l1 = np.cross(vp1[0], vp1[1])
    l2 = np.cross(vp2[0], vp2[1])
    w_inv = K @ K.T
    angle = l1 @ w_inv @ l2
    angle /= np.sqrt(l1 @ w_inv @ l1)
    angle /= np.sqrt(l2 @ w_inv @ l2)
    return np.rad2deg(np.arccos(angle))

'''
COMPUTE_ROTATION_MATRIX_BETWEEN_CAMERAS
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''
def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    # see https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    K_inv = np.linalg.inv(K)
    vp1 = np.hstack((vanishing_points1, np.ones((3,1))))
    vp2 = np.hstack((vanishing_points2, np.ones((3,1))))
    d1 = K_inv @ vp1.T 
    d1 /= np.linalg.norm(d1, axis=0)
    d2 = K_inv @ vp2.T 
    d2 /= np.linalg.norm(d2, axis=0)
    M = d2 @ d1.T
    u,s,v = np.linalg.svd(M)
    R = u @ v
    # print(R @ R.T) see if the matrix is orthogonal
    return R


if __name__ == '__main__':
    # Part A: Compute vanishing points
    v1 = compute_vanishing_point(np.array([[674,1826],[2456,1060],[1094,1340],[1774,1086]]))
    v2 = compute_vanishing_point(np.array([[674,1826],[126,1056],[2456,1060],[1940,866]]))
    v3 = compute_vanishing_point(np.array([[1094,1340],[1080,598],[1774,1086],[1840,478]]))

    v1b = compute_vanishing_point(np.array([[314,1912],[2060,1040],[750,1378],[1438,1094]]))
    v2b = compute_vanishing_point(np.array([[314,1912],[36,1578],[2060,1040],[1598,882]]))
    v3b = compute_vanishing_point(np.array([[750,1378],[714,614],[1438,1094],[1474,494]]))

    # Part B: Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    # print(vanishing_points)
    print("Intrinsic Matrix:\n",compute_K_from_vanishing_points(vanishing_points))

    K_actual = np.array([[2448.0, 0, 1253.0],[0, 2438.0, 986.0],[0,0,1.0]])
    print()
    print("Actual Matrix:\n", K_actual)
    # Part D: Estimate the angle between the box and floor
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094,1340],[1774,1086],[1080,598],[1840,478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2], K_actual)
    print()
    print("Angle between floor and box:", angle)

    # Part E: Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print("Rotation between two cameras:\n", rotation_matrix)
    z,y,x = mat2euler(rotation_matrix)
    print()
    print("Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi))
    print("Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi))
    print("Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi))
