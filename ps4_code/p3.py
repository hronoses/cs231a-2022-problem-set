import numpy as np
import pdb

# TODO fill out the following class, reusing your implementation from the 
# prior problems
class Q3_solution(object):

  @staticmethod
  def system_matrix():
    """ 
    Output:
      A: 6x6 numpy array for the system matrix.
    """
    A = np.eye(6)
    A[:3, 3:] = np.eye(3) * 0.1
    A[3:, 3:] *= 0.8
    return A

  @staticmethod
  def process_noise_covariance():
    """ 
    Output:
      Q: 6x6 numpy array for the covariance matrix.
    """
    # Hint: this is the same as in part A
    Q = np.zeros((6, 6))
    Q[3:, 3:] = np.eye(3) * 0.05
    return Q

  @staticmethod
  def observation_noise_covariance():
    """ 
    Output:
      R: 3x3 numpy array for the covariance matrix.
    """
    sigma = np.diag([0.005, 0.005, 0.01])*0.1
    return sigma

  def KF(self, observations, mu_0, sigma_0, remove_outliers=False):
    """ Implement Kalman filtering 
    Input:
      observations: (N,3) numpy array, the sequence of observations. From T=1.
      mu_0: (6,) numpy array, the mean of state belief after T=0
      sigma_0: (6,6) numpy array, the covariance matrix for state belief after T=0.
      remove_outliers: bool, whether to remove outliers
    Output:
      state_mean: (N+1,6) numpy array, the filtered mean state at each time step, starting at T=0 
      state_sigma: (N+1,6,6) numpy array, the filtered state covariance at each time step, starting at T=0 
      predicted_observation_mean: (N,2) numpy array, the mean of predicted observations. Start from T=1
      predicted_observation_sigma: (N,2,2) numpy array, the covariance matrix of predicted observations. Start from T=1
    """
    A = self.system_matrix()
    Q = self.process_noise_covariance()
    R = self.observation_noise_covariance()
    state_mean = [mu_0]
    state_sigma = [sigma_0]
    predicted_observation_mean = []
    predicted_observation_sigma = []
    self.deviations = np.zeros(observations.shape[0])
    for i, ob in enumerate(observations):
        mu_bar_next = A @ state_mean[-1] 
        sigma_bar_next = A @ state_sigma[-1] @ A.T + Q
        C = np.hstack([np.eye(3), np.zeros((3,3))])
        kalman_gain_numerator = sigma_bar_next @ C.T
        kalman_gain_denominator = C @ sigma_bar_next @ C.T + R
        kalman_gain = kalman_gain_numerator @ np.linalg.inv(kalman_gain_denominator)
        expected_observation = mu_bar_next[:3]
        self.deviations[i] = np.sum(np.abs(expected_observation - ob))
        # outlier = x coordinate does not move
        outlier = False
        if 0 < i < observations.shape[0] - 1:
           outlier = np.linalg.norm(observations[i] - observations[i + 1]) < 0.0001 or \
                    np.linalg.norm(observations[i] - observations[i - 1]) < 0.0001
        if remove_outliers and outlier: # outliers
            kalman_gain *= 0
        mu_next = mu_bar_next + kalman_gain @ (ob - expected_observation)
        sigma_next = sigma_bar_next - kalman_gain @ C @ sigma_bar_next
        state_mean.append(mu_next)
        state_sigma.append(sigma_next)
        predicted_observation_mean.append(expected_observation)
        predicted_observation_sigma.append(kalman_gain_denominator)
    return state_mean, state_sigma, predicted_observation_mean, predicted_observation_sigma


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from plot_helper import draw_2d, draw_3d
    # part_b_data = True
    # filter_outliers = False
    part_b_data = False
    filter_outliers = True
    
    solution = Q3_solution()
    
    if part_b_data:
        print('part B. Loading data...')
        state_positions = np.load('data/Q3B_data/Q3B_positions_gt.npy')
        observations = np.load('./data/Q3B_predictions.npy')
    else:
        print('part D. Loading data...')
        state_positions = np.load('data/Q3D_data/Q3D_positions_gt.npy')
        observations = np.load('./data/Q3D_predictions.npy')
    state_0 = np.concatenate([state_positions[0], np.zeros(3)])
    sigma_0 = np.eye(6)*0.01
    sigma_0[3:,3:] = 0.0
    state_mean, state_sigma, predicted_observation_mean, predicted_observation_sigma = \
        solution.KF(observations[1:], state_0, sigma_0, filter_outliers)

    state_mean = np.array(state_mean)
    dev = np.linalg.norm(state_positions-state_mean[:,:3], axis=1)
    err = np.linalg.norm(state_positions-observations[:,:3], axis=1)
    print(np.mean(solution.deviations))
    print(np.mean(dev))
    print(np.mean(err))
    # 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(state_mean[:,0], state_mean[:,1], state_mean[:,2], c='C1')
    ax.scatter(state_positions[:,0], state_positions[:,1], state_positions[:,2], c='C0')
    # ax.scatter(observations[:,0], observations[:,1], observations[:,2], c='C0')
    plt.show()

    if not part_b_data:
      # 2d plotting
      fig = plt.figure()
      plt.scatter(state_positions[:,0], state_positions[:,2], c='C0')
      state_mean = np.array(state_mean)
      plt.scatter(state_mean[:,0], state_mean[:,2], c='C1')
      plt.show()

      fig = plt.figure()
      plt.scatter(state_positions[:,1], state_positions[:,2], c='C0')
      state_mean = np.array(state_mean)
      plt.scatter(state_mean[:,1], state_mean[:,2], c='C1')
      plt.show()
      
      if filter_outliers:
          def projection(points_3d):
              intrinsic = np.array([[500,0,320],[0,500,240],[0,0,1]])
              points_proj = np.dot(points_3d, intrinsic.transpose())
              points_proj = points_proj[:,:2]/points_proj[:,2:3]
              return points_proj

          gt_state_proj = projection(state_positions)
          detection_proj = projection(observations)
          filtered_state_proj = projection(np.array(state_mean)[:,:3])

          fig = plt.figure()
          ax = fig.add_subplot(111)
          ax.scatter(gt_state_proj[:,0], gt_state_proj[:,1], s=4, label='True projection')
          ax.scatter(detection_proj[:,0], detection_proj[:,1], s=4, label='Detected projection')
          ax.scatter(filtered_state_proj[:,0], filtered_state_proj[:,1], s=4, label='Filtered projection')
          plt.xlim([0,640])
          plt.ylim([0,480])
          plt.legend()
          plt.gca().invert_yaxis()
          plt.show()



