import numpy as np
from numba import jit, njit, prange
import multiprocessing
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy

def translate_policy(policy, nActions, stationary=True):
    if not stationary: # shape H,S
        return np.array([np.eye(nActions)[np.array(pi_h)]
                        for pi_h in policy])
    # translates deterministic policy to pi_{s,a}
    return np.eye(nActions)[np.array(policy)]

def c_star(pi_star, pi):
    dens = pi_star/pi
    dens[np.isnan(dens)] = 1
    return np.max(dens)

def collect_sample(nsamples, mdp, pi_b, horizon, stationary=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dataset = []
    for _ in range(nsamples):
        traj = mdp.generate_trajectory(pi_b, horizon, stationary)
        dataset.append(traj)
    dataset = np.array(dataset)
    # x, a, x', r
    return dataset

def getSamplesMultiProc(samples, mdp, pi_b, horizon, start_seed=0):
    nprocs = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=nprocs, mp_context=multiprocessing.get_context('fork')) as executor:
        future = executor.map(collect_sample, [int(samples/nprocs) for i in range(nprocs)], repeat(copy.deepcopy(mdp)), 
                              repeat(copy.deepcopy(pi_b)), repeat(horizon), [i+start_seed for i in range(nprocs)])
    dataset = np.vstack(list(future))
    return dataset


# Gets counts of occupancies of state-action tuples in dataset
#    optional parameter burnin if one wants to only take counts past mixing time
def getN_sa(dataset, nStates, nActions, burnin=0, reshape=True):
    N_sa = np.zeros((nStates, nActions))
    if reshape:
        resdata = dataset[:, burnin:, :].reshape(dataset[:, burnin:, :].shape[0]*dataset[:, burnin:, :].shape[1], 
                                                     dataset[:, burnin:, :].shape[2])
    else:
        resdata = np.copy(dataset)
    for s,a,sp,r in resdata:
        N_sa[int(s),int(a)] += 1
    return N_sa

def getOcc(dataset, nStates, nActions):
    return (getN_sa(dataset, nStates, nActions)
            /dataset.shape[0]/dataset.shape[1])

def getR_sa(dataset, nStates, nActions):
    N_sa = getN_sa(dataset, nStates, nActions)
    R_sa = np.zeros((nStates, nActions))
    for s,a,sp,r in dataset.reshape(dataset.shape[0]*dataset.shape[1], 
                                                     dataset.shape[2]):
        R_sa[int(s),int(a)] += r
    R_sa = (R_sa/N_sa)
    R_sa[np.isnan(R_sa)] = 0
    R_sa[np.isinf(R_sa)] = 0
    return R_sa


def getPhat(dataset, nStates, nActions):
    N_sa = getN_sa(dataset, nStates, nActions)
    Phat = np.zeros((nActions, nStates, nStates))
    for s,a,sp,r in dataset.reshape(dataset.shape[0]*dataset.shape[1], 
                                                     dataset.shape[2]):
        Phat[int(a),int(s),int(sp)] += 1
    Phat = (Phat/N_sa.T[...,None])
    Phat[np.isnan(Phat)] = 0
    Phat[np.isinf(Phat)] = 0
    return Phat

def getN_asp(dataset, nStates, nActions, burnin=0, reshape=True):
    N_asp = np.zeros((nActions, nStates, nStates))
    if reshape:
        resdata = dataset[:, burnin:, :].reshape(dataset[:, burnin:, :].shape[0]*dataset[:, burnin:, :].shape[1], 
                                                     dataset[:, burnin:, :].shape[2])
    else:
        resdata = np.copy(dataset)
    for s,a,sp,r in resdata:
        N_asp[int(a), int(s), int(sp)] += 1
    return N_asp

def getR_asp(dataset, nStates, nActions):
    N_asp = getN_asp(dataset, nStates, nActions)
    R_asp = np.zeros((nActions,nStates,nStates))
    for s,a,sp,r in dataset.reshape(dataset.shape[0]*dataset.shape[1], 
                                                     dataset.shape[2]):
        R_asp[int(a), int(s), int(sp)] += r
    R_asp = (R_asp/N_asp)
    R_asp[np.isnan(R_asp)] = 0
    R_asp[np.isinf(R_asp)] = 0
    return R_asp

def eval_pi(pi_e, P, R_sa, nStates, horizon):
    Pb_spsa = P.transpose(2,1,0)
    V = np.zeros((horizon+1, nStates))
    def recursiveV(h, s, pi_e, R_sa, Pb_spsa):
        if h == horizon:
            return 0
        elif V[h,s] != 0:
            return V[h,s]
        else:
            Vhp = np.array([recursiveV(h+1,sp, pi_e, R_sa, Pb_spsa) for sp in range(nStates)])
            V[h+1,:] = Vhp
            return pi_e[s,:].T @ (R_sa[s,:] + Pb_spsa[:,s,:].T @ Vhp)
    V = np.zeros((horizon+1, nStates))
    return recursiveV(0, 0, pi_e, R_sa, Pb_spsa)

