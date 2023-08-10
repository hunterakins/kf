import numpy as np
from matplotlib import pyplot as plt
from signal_proc.kalman_filter import simple_kalman as sk

"""
Description:
Test simple_kalman code

Date:
7/6/2022

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

def twod_motion_track():
    """ 
    Model is noisy observations of position
    of object moving at constant speed
    """
    vx = 1. 
    vy = 2.
    F = np.identity(4)
    F[0,2] = 1 # x \to x + vx delta t , delta t = 1
    F[1, 3] = 1
    sigma_pos = .01
    sigma_vel = .01

    Q = np.identity(4)
    Q[0,0] *= np.square(sigma_pos)
    Q[1,1] *= np.square(sigma_pos)
    Q[2,2] *= np.square(sigma_vel)
    Q[3,3] *= np.square(sigma_vel)
    dm = sk.DynamicalModel(F, Q)
    H = np.zeros((2,4))
    H[0,0] = 1
    H[1,1] = 1
    sigma_n = .5
    R = np.square(sigma_n)*np.identity(2)
    om = sk.ObservationModel(H, R)
    kf = sk.KalmanFilter(dm, om)
    N = 100

    x0_t = np.array([.2, 1, vx, vy]).reshape(4,1)
    true_state_list, obs_list = kf.sim_obs(x0_t, N)


    x0 = np.zeros((4, 1))
    P0 = np.sqrt(4) * np.identity(4)
    kf.initialize(x0, P0)
    kf.add_data(obs_list)
    kf.run_filter()
    plt.figure()
    true_x, true_y = [x[0,0] for x in true_state_list[:]] , [x[1,0] for x in true_state_list[:]]
    true_vx, true_vy = [x[2,0] for x in true_state_list[:]] , [x[3,0] for x in true_state_list[:]]
    obs_x, obs_y = [x[0,0] for x in obs_list[:]] , [x[1,0] for x in obs_list[:]]
    plt.plot(true_x, true_y, 'b.')
    plt.plot(obs_x, obs_y, 'k*')

    errs = []
    for i in range(N):
        est_state = kf.x_hat_list[i]
        true_state = true_state_list[i+1]
        err = abs(est_state - true_state)
        errs.append(err)
        #plt.plot(true_state[0,0], true_state[1,0], 'b+')
        #plt.plot(est_state[0,0], est_state[1,0], 'r+')

    est_x, est_y = [x[0,0] for x in kf.x_hat_list] , [x[1,0] for x in kf.x_hat_list]
    est_vx, est_vy = [x[2,0] for x in kf.x_hat_list] , [x[3,0] for x in kf.x_hat_list]
    plt.plot(est_x[1:], est_y[1:], 'r+')
    plt.legend(['True trajectory', 'Observations', 'Estimated trajectory'])

    plt.figure()
    plt.plot(est_x, 'r+')
    plt.plot(true_x, 'b.')
    plt.plot(obs_x, 'k*')
    

    plt.figure()
    plt.plot(true_vx, 'b')
    plt.plot(true_vy, 'b')
    plt.plot(est_vx, 'r')
    plt.plot(est_vy, 'r')
    plt.legend(['vx', 'vy', 'est vx', 'est vy'])

    x_cov = [np.sqrt(x[0,0]) for x in kf.P_hat_list]
    
    plt.figure() 
    plt.plot(range(N), [x[0] for x in errs])
    plt.plot(range(N), [x[1] for x in errs])
    plt.plot(range(N), [sigma_n]*N)
    plt.plot(range(N), x_cov)
    plt.legend(['x err', 'y err', 'Obs noise std.', 'posterior std'])
    plt.xlabel('Step index')

    plt.ylabel('$||\widehat{x_{k | k}} - x_{k}||_{2}$')

    plt.show()

if __name__ == '__main__':
    twod_motion_track()        

    

