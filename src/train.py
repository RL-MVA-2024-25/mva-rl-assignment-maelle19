from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle
import random
from xgboost import XGBRegressor

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


class ProjectAgent:

    def __init__(self):
        self.model = XGBRegressor()  # Remplacement par XGBoost
        
    def act(self, observation, use_random=False):
        if use_random:
            a = random.randint(0, env.action_space.n-1)
        else:
            a = self.greedy_action(self.model, observation, env.action_space.n)
        return a
    
    def train(self):
        size_collect = 20
        S, A, R, S2 = self.collect_samples(env, size_collect)
        iterations = 700
        Qfunctions = self.rf_fqi(S, A, R, S2, iterations, nb_actions=env.action_space.n, gamma=0.9, size_collect=size_collect)
        self.model = Qfunctions[-1]
        self.save("model_GB_2")
    
    def collect_samples(self, env, horizon, Q=None, eps=0.1, s0=None):
        if s0 is None:
            s, _ = env.reset()
        else:
            s = s0
        S, A, R, S2 = [], [], [], []

        for _ in range(horizon):
            if Q is None:
                a = env.action_space.sample()
            else:
                r = np.random.rand()
                if r > eps:
                    a = self.greedy_action(Q, s, env.action_space.n)
                else:
                    a = env.action_space.sample()

            s2, r, _, _, _ = env.step(a)
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            s = s2

        S = np.array(S)
        A = np.array(A).reshape((-1, 1))
        R = np.array(R)
        S2 = np.array(S2)
        return S, A, R, S2
    
    def rf_fqi(self, S, A, R, S2, iterations, nb_actions, gamma, size_collect):
        Qfunctions = []
        SA = np.append(S, A, axis=1)
        States, Actions, States2, Rewards = S.copy(), A.copy(), S2.copy(), R.copy()

        for iter in range(iterations):
            if iter == 0:
                value = R.copy()
            else:
                s0 = None if iter % 10 == 0 else S[-1]
                S_1, A_1, R_1, S2_1 = self.collect_samples(env, size_collect, Q=Qfunctions[-1], s0=s0)
                States = np.append(States, S_1, axis=0)
                Actions = np.append(Actions, A_1, axis=0)
                States2 = np.append(States2, S2_1, axis=0)
                Rewards = np.append(Rewards, R_1, axis=0)
                SA = np.append(States, Actions, axis=1)

            if iter > 0:
                Q2 = np.zeros((States.shape[0], nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2 * np.ones((States.shape[0], 1))
                    S2A2 = np.append(States2, A2, axis=1)
                    Q2[:, a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2, axis=1)
                value = Rewards + gamma * max_Q2

            Q = XGBRegressor()
            Q.fit(SA, value)
            Qfunctions.append(Q)
        return Qfunctions

    def greedy_action(self, Q, s, nb_actions):
        Qsa = []
        for a in range(nb_actions):
            sa = np.append(s, a).reshape(1, -1)
            Qsa.append(Q.predict(sa))
        return np.argmax(Qsa)

    def save(self, path="model_GB_2"):
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)

    def load(self):
        path = "model_GB_2"
        with open(path, 'rb') as file:
            self.model = pickle.load(file)