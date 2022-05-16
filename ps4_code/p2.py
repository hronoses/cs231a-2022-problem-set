import numpy as np
from p1 import Q1_solution


class Q2_solution(Q1_solution):

  @staticmethod
  def observation(x):
    """ Implement Q2A. Observation function without noise.
    Input:
      x: (6,) numpy array representing the state.
    Output:
      obs: (3,) numpy array representing the observation (u,v,d).
    Note:
      we define disparity to be possitive.
    """
    # Hint: this should be similar to your implemention in Q1, but with two cameras
     # Hint: you should use the camera intrinsics here
    K = np.array([[500, 0, 320],
                [0, 500, 240],
                [0, 0, 1]])
    # since we have equal fx fy it does not metter along which axis we make camera displacement
    b = np.array([0.2, 0, 0])

    obs_left = K @ x[:3]
    obs_right = K @ (x[:3] + b)
    obs_left /= obs_left[-1]
    obs_right /= obs_right[-1]
    disparity = obs_left[0] - obs_right[0]
    obs_left[-1] = np.abs(disparity)
    return obs_left

  @staticmethod
  def observation_state_jacobian(x):
    """ Implement Q2B. The jacobian of observation function w.r.t state.
    Input:
      x: (6,) numpy array, the state to take jacobian with.
    Output:
      H: (3,6) numpy array, the jacobian H.
    """
    b = 0.2 
    H = np.zeros((3,6))
    H[0, 0] = 500 / x[2] 
    H[0, 2] =  - 500 * x[0] / (x[2] ** 2) 
    H[1, 1] = 500 / x[2] 
    H[1, 2] =  - 500 * x[1] / (x[2] ** 2)
    H[2, 2] =  - 500 * b / (x[2] ** 2)
    return H

  @staticmethod
  def observation_noise_covariance():
    """ Implement Q2C here.
    Output:
      R: (3,3) numpy array, the covariance matrix for observation noise.
    """
    R = np.eye(3) * 5
    return R


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from plot_helper import draw_2d, draw_3d

    np.random.seed(315)
    solution = Q2_solution()
    states, observations = solution.simulation()
    # plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(states[:,0], states[:,1], states[:,2], c=np.arange(states.shape[0]))
    plt.show()

    fig = plt.figure()
    plt.scatter(observations[:,0], observations[:,1], c=np.arange(states.shape[0]), s=4)
    plt.xlim([0,640])
    plt.ylim([0,480])
    plt.gca().invert_yaxis()
    plt.show()

    observations = np.load('./data/Q2D_measurement.npy')
    filtered_state_mean, filtered_state_sigma, predicted_observation_mean, predicted_observation_sigma = \
        solution.EKF(observations)
    # plotting
    true_states = np.load('./data/Q2D_state.npy')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(true_states[:,0], true_states[:,1], true_states[:,2], c='C0')
    for mean, cov in zip(filtered_state_mean, filtered_state_sigma):
        draw_3d(ax, cov[:3,:3], mean[:3])
    ax.view_init(elev=10., azim=45)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(observations[:,0], observations[:,1], s=4)
    for mean, cov in zip(predicted_observation_mean, predicted_observation_sigma):
        draw_2d(ax, cov[:2,:2], mean[:2])
    plt.xlim([0,640])
    plt.ylim([0,480])
    plt.gca().invert_yaxis()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(observations[:,0]-observations[:,2], observations[:,1], s=4)
    for mean, cov in zip(predicted_observation_mean, predicted_observation_sigma):
        right_mean = mean[:2] - np.array([mean[2], 0])
        right_cov = cov[:2,:2] 
        right_cov[0,0] += cov[2,2] - 2 * cov[0, 2] 
        right_cov[0,1] -= cov[1, 2] 
        right_cov[1,0] -= cov[1, 2] 
        draw_2d(ax, right_cov, right_mean)
    plt.xlim([0,640])
    plt.ylim([0,480])
    plt.gca().invert_yaxis()
    plt.show()




