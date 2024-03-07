import core_mdp
import numpy as np
from tqdm import tqdm

import scipy
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import SR1, BFGS
from scipy.optimize import minimize

#------------------------------------------------------------------------------------
#   Importance sampling
#------------------------------------------------------------------------------------

def IS(dataset, gamma, horizon, pihat, pi_e):
    rets = core_mdp.calc_returns(dataset, gamma, horizon)
    weighted_data = 0
    for traj,ret in zip(dataset,rets):
        rho = 1
        for x,a,xp,r in traj:
            rho *= (pi_e[int(x), int(a)] / pihat[int(x), int(a)])
        weighted_data += ret * rho
    return weighted_data / len(dataset)

def WIS(dataset, gamma, horizon, pihat, pi_e):
    rets = core_mdp.calc_returns(dataset, gamma, horizon)
    weighted_data = 0
    is_sum = 0
    for traj,ret in zip(dataset,rets):
        rho = 1
        for x,a,xp,r in traj:
            rho *= (pi_e[int(x), int(a)] / pihat[int(x), int(a)])
        weighted_data += ret * rho
        is_sum += rho
    return weighted_data / is_sum

#------------------------------------------------------------------------------------
#   FQE and helper functions
#------------------------------------------------------------------------------------

def fitted_q_update(f, pi_e, dataset, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions
    data = dataset.reshape((dataset.shape[0]*dataset.shape[1],4))
    regression_data = np.array([[x, a, r + mdp.gamma * (pi_e[int(xp)] @ f[int(xp),:])] for x,a,xp,r in data])
    Tf_hat = np.zeros((nStates, nActions))
    for x,a,y in regression_data:
        Tf_hat[int(x), int(a)] += y
    idx, count = np.unique(data[:,:2], axis=0, return_counts=True)
    for i,[x,a] in enumerate(idx):
        Tf_hat[int(x),int(a)] /= count[i]
    return Tf_hat

def fitted_q_evaluation(pi_e, dataset, horizon, mdp):
    Qhat = np.zeros((mdp.n_states, mdp.n_actions))
    for k in tqdm(range(horizon)):
        newQ = fitted_q_update(Qhat, pi_e, dataset, mdp)
        #trueNewQ = bellman_eval_update(Qhat, np.array([pi_e,pi_e]))
        #print("Squared error: " + str(((newQ - trueNewQ)**2).sum()))
        Qhat = newQ
    return Qhat
