import numpy as np
from matplotlib import pyplot as plt

"""
Description:
Kalman filter class
Follows notation on https://en.wikipedia.org/wiki/Kalman_filter


Date:
7/6/2022

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

class DynamicalModel:
    def __init__(self, F, Q):
        self.F = F
        self.Q = Q

    def _update_Q(self, Q):
        """ If noise covariance changes from sample to sample
        you can update it 
        """
        self.Q = Q
        return

    def _update_F(self, F):
        """ If state-transition model changes """
        self.F = F
        return

class ObservationModel:
    def __init__(self, H, R):
        self.H = H
        self.R = R
        
    def _update_H(self, H):
        self.H = H

    def _update_R(self, R):
        self.R = R
    
class KalmanFilter:
    def __init__(self, dynamical_model, observation_model):
        self.dm = dynamical_model
        self.om = observation_model
        self.P_hat = None # most up to date estimation
        self.x_hat = None # most up to date estimation
        self.P_hat_list = [] # save all estimted covs
        self.x_hat_list = [] # save all estimted states
        self.z_list = None # observations

    def initialize(self, x0, prior_cov):
        self.P_hat = prior_cov
        self.x_hat = x0

    def _predict(self):
        x_pred = self.dm.F@self.x_hat
        P_pred = self.dm.F@self.P_hat@self.dm.F.T + self.dm.Q
        self.x_pred = x_pred
        self.P_pred = P_pred
        return x_pred, P_pred
        
    def _update(self, z):
        """ Given observation zk, update xhat and chat  """
        x_pred, P_pred = self._predict()
        H = self.om.H
        ytilde = z - H@x_pred # innovation
        S = H@P_pred@H.T + self.om.R
        # Kalman gain K = P_pred@H.T@S_inv
        S_inv = np.linalg.inv(S)
        delta_x = P_pred@H.T@np.linalg.solve(S, ytilde)
        x_update = x_pred + delta_x
        P_update = P_pred - P_pred@H.T@(S_inv)@H@P_pred
        self.x_hat = x_update
        self.P_hat = P_update

    def sim_obs(self, x0, N):
        """
        Simulate observations 
        """
        R = self.om.R
        L = np.linalg.cholesky(R)
        obs_noise = L@np.random.randn(R.shape[0], N)

        Q = self.dm.Q
        LQ = np.linalg.cholesky(Q)
        state_noise = LQ@np.random.randn(x0.size,N)
        
        obs_list = []
        state_list = []
        n_y = R.shape[1]
        for i in range(N):
            x0 = self.dm.F @ x0 + state_noise[:,i].reshape(x0.size,1)
            z0 = self.om.H@x0 + obs_noise[:,i].reshape(n_y,1)
            obs_list.append(z0)
            state_list.append(x0)
        #state_list.append(x0)
        return state_list, obs_list

    def add_data(self, z_list):
        """
        Add data to the filter
        """
        self.z_list = z_list
        return

    def run_filter(self):
        num_obs = len(self.z_list)
        for i in range(num_obs):
            zk = self.z_list[i]
            self._update(zk)
            self.x_hat_list.append(self.x_hat.copy())
            self.P_hat_list.append(self.P_hat.copy())
        return 

class UKIHyperParams: 
    def __init__(self, kappa, alpha, beta):
        self.kappa = kappa
        self.alpha=  alpha
        self.beta = beta

class UnscentedKalmanFilter(KalmanFilter):
    def __init__(self, dynamical_model, observation_model, uki_params):
        super().__init__(dynamical_model, observation_model)
        self.uki_params = uki_params
        self.nu_list = []
        self.resid_norm_list = []

    def sim_obs(self, x0, N):
        """
        Simulate observations 
        """
        R = self.om.R
        L = np.linalg.cholesky(R)
        obs_noise = L@np.random.randn(R.shape[0], N)

        Q = self.dm.Q
        LQ = np.linalg.cholesky(Q)
        state_noise = LQ@np.random.randn(x0.size,N)
        
        obs_list = []
        state_list = []
        n_y = R.shape[1]
        for i in range(N):
            x0 = self.dm.F @ x0 + state_noise[:,i].reshape(x0.size,1)
            z0 = self.om.H(x0) + obs_noise[:,i].reshape(n_y,1)
            obs_list.append(z0)
            state_list.append(x0)
        return state_list, obs_list

    def _gen_sigma_pts(self, x_hat, sqrt_P_hat):
        """ Use estimates to generate the sigma point
        I follow notation in Huang, Schneider, and Stuart, """
        L = sqrt_P_hat.shape[1]
        alpha = self.uki_params.alpha
        kappa = self.uki_params.kappa
        lam = np.square(alpha)*(L + kappa) - L

        c = np.sqrt(L + lam)
        x_hat_plus = np.zeros((x_hat.size, L))
        x_hat_minus = np.zeros((x_hat.size, L))

        for i in range(L):
            sig = c*sqrt_P_hat[:,i]
            x_hat_plus[:,i] = x_hat[:,0] + sig
            x_hat_minus[:,i] = x_hat[:,0] - sig
        return x_hat_plus, x_hat_minus 
        
    def _update(self, z, debug_mode=False):

        n_y = z.size
        """ Generate sigma points """
        x_hat, P_hat = self.x_hat, self.P_hat
        eigs, vec_mat = np.linalg.eigh(P_hat)
        i = 0
        total_pow = np.sum(eigs)
        while np.sum(eigs[-(i+1):]) < .99*total_pow:
            i += 1
        i += 1

        if debug_mode == True:
            plt.figure()
            plt.plot(eigs/total_pow)
            plt.plot(eigs[-i:]/total_pow)
            plt.plot([.01] *eigs.size, 'k--')
            plt.figure()
            plt.plot(vec_mat[:,:])



        """ Compute weights used """
        n_x = self.dm.F.shape[0]
        L = n_x
        alpha = self.uki_params.alpha
        kappa = self.uki_params.kappa
        b = self.uki_params.beta
        lam = np.square(alpha)*(L + kappa) - L
        w0m = lam / (L + lam) 
        w0c = lam / (L + lam) + (1 - np.square(alpha) + b)
        wjm = 1 / (2*(L + lam) )
        wjc = wjm
        sqrt_P_hat = np.linalg.cholesky(P_hat)
       
        x_hat_plus, x_hat_minus = self._gen_sigma_pts(x_hat, sqrt_P_hat)
        x_hat_sigma = np.hstack((x_hat_plus, x_hat_minus))


        """ Predict the sigma points """
        x_pred_sigma = self.dm.F@x_hat_sigma
        x_pred_sigma0 = self.dm.F@x_hat

        """ Use them to form the predictive estimate x_hat_{k | k-1} 
        and the predictive covariance P_hat_{k | k-1} """
        x_hat_pred = w0m*x_pred_sigma0 + wjm*np.sum(x_pred_sigma, axis=1)[:, np.newaxis]
        x_diffs = x_pred_sigma - x_hat_pred 

        P_hat_pred = np.zeros((n_x, n_x))
        for i in range(2*L):
            P_hat_pred += wjc*np.outer(x_diffs[:,i], x_diffs[:,i])
        x_diff0 = x_pred_sigma0 - x_hat_pred
        P_hat_pred += w0c*x_diff0@x_diff0.T 
        P_hat_pred += self.dm.Q
        
    
        """ Move the sigma points through the model to get prediction of \hat{y}_{k | k-1} 
        and covs """
        H = self.om.H
        y_hat_sigma = np.zeros((n_y, 2*n_x))
        for i in range(2*L):
            y_hat_sigma[:,i] = np.squeeze(H(x_pred_sigma[:,i].reshape(n_x, 1)))
        y_hat_sigma0 = H(x_pred_sigma0)

        """ Diagnostic matrix """
        nu = np.zeros((n_y, n_x)) 
        for i in range(L):
            nu[:,i] = y_hat_sigma[:,i] + y_hat_sigma[:, i + n_x] - 2*y_hat_sigma0[:,0]

        nu_vec = np.linalg.norm(nu, axis=0)
        #print('nu vec shape', nu_vec.shape, 'should be 1, nx')
        #print(nu_vec)
        if debug_mode == True:
            plt.figure()
            plt.plot(sqrt_P_hat)

            plt.figure()
            plt.plot(x_hat_sigma)
            plt.suptitle('Sigma points')
            plt.figure()
            plt.plot(y_hat_sigma)
            plt.plot(z, 'k')
            plt.suptitle('yhat sigma')
            plt.show()


        y_hat = w0m*y_hat_sigma0 + wjm*np.sum(y_hat_sigma, axis=1)[:, np.newaxis]

        y_hat_diffs = y_hat_sigma - y_hat
        y_hat_diff0 = y_hat_sigma0 - y_hat
       
        P_xy = np.zeros((n_x, n_y))
        P_yy = np.zeros((n_y, n_y))
        for i in range(2*L):
            xdiff = x_diffs[:,i].reshape((n_x, 1))
            ydiff = y_hat_diffs[:,i].reshape((n_y, 1))
            P_xy += xdiff @ ydiff.T
            P_yy += ydiff @ ydiff.T

        P_xy *= wjc
        P_yy *= wjc
        P_xy += w0c*(x_diff0@y_hat_diff0.T)
        P_yy += w0c*(y_hat_diff0@y_hat_diff0.T)
        P_yy += self.om.R

        """ Now use observation to update the covariances """
        ytilde = z - y_hat
        x_hat = x_hat_pred + P_xy @ np.linalg.solve(P_yy, ytilde)
        P_yy_inv = np.linalg.inv(P_yy) 
        P_hat = P_hat_pred - P_xy@P_yy_inv@P_xy.T
        self.x_hat = x_hat
        self.P_hat = P_hat
        self.nu = nu_vec
        self.resid_norm = np.linalg.norm(ytilde)
        return

    def run_filter(self):
        num_obs = len(self.z_list)
        for i in range(num_obs):
            zk = self.z_list[i]
            self._update(zk)
            self.x_hat_list.append(self.x_hat.copy())
            self.P_hat_list.append(self.P_hat.copy())
            self.nu_list.append(self.nu.copy())
            self.resid_norm_list.append(self.resid_norm.copy())
        return 
