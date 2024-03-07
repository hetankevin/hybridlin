import numpy as np

class MDP(object):
    def __init__(self, P, R, x_dist, gamma=1):
        self.P = P  # A by S by S
        self.R = R

        self.n_actions = P.shape[0]
        self.n_states = P.shape[1]
        self.gamma = gamma

        self.x_dist = x_dist

        self.reset()

    def reset(self):
	    self.state = np.random.choice(self.n_states, p=self.x_dist)
	    self.done = False
	    return self.state

    def step(self, a):
        x = self.state
        xp = np.random.choice(self.n_states, p=self.P[a,x])
        r = self.R[x, a]
        self.state = xp
        return self.state, r

    def generate_trajectory(self, pi_b, horizon, stationary=True):
        # stationary: pi_b is shaped (s, a)
        # else: pi_b is shaped (h, s, a)
        self.reset()
        traj = []
        for t in range(horizon):
            x = self.state
            if stationary:
                a = np.random.choice(self.n_actions, p=pi_b[x])
            else:
                a = np.random.choice(self.n_actions, p=pi_b[t, x, :])
            xp, r = self.step(a)
            traj.append([x,a,xp,r])
        return np.array(traj)

    def bellman_eval_update(self, f, pi):
        R = self.R
        P = self.P
        nStates = self.n_states
        nActions = self.n_actions
        gamma = self.gamma
        Tf = np.zeros(f.shape)
        for s in range(nStates):
            for a in range(nActions):
                f_pi_avg = np.array([pi[xp] @ f[xp, :] for xp in range(nStates)])
                Tf[s,a] += P[a, s] @ (R[s, a] + gamma * f_pi_avg) 
        return Tf

    def bellman_eval(self, pi, horizon):
        Q = np.zeros((self.n_states, self.n_actions))
        for k in range(horizon):
            Q = self.bellman_eval_update(Q, pi)
        return Q

    def get_value(self, Q, pi):
        V = np.array([Q[x] @ pi[x] for x in range(self.n_states)])
        avgV = V @ self.x_dist
        return V, avgV


def collect_sample(nsamples, mdp, pi_b, horizon):
    dataset = []
    for _ in range(nsamples):
        traj = mdp.generate_trajectory(pi_b, horizon)
        dataset.append(traj)
    dataset = np.array(dataset)
    # x, a, x', r
    return dataset


def calc_returns(data, gamma, horizon):
    rewards = data[:,:,-1]
    g = np.array([gamma**t for t in range(horizon)])
    return (rewards * g).sum(axis=1)