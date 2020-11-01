import gym
import numpy as np 
import random


class Sarsa:

    def run(self, 
            episodes, 
            steps, 
            evnironment):

        # Load Environment and Q-table structure
        env = gym.make(evnironment)

        # Random init of Q
        Q = np.random.randint(0, env.action_space.n-1, 
                              size=(env.observation_space.n, 
                                    env.action_space.n))

        epsilon = 0.10
        alpha = 0.7
        gamma = 0.9

        Gvector = np.arange(episodes)

        # Sarsa Algorithm
        for i in range(episodes):
            # Reset environment
            s = env.reset()

            G = 0

            # Choose init action using epsilon greedy
            if (random.random() < epsilon):
                # Random action 
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[s,:]) 

            index = 0
            done = False
            while (done == False) and (index < steps) :
                env.render()

                # Take action a 
                # Get new state & reward from environment
                s1, r, done, diag = env.step(a)

                # Choose action a1 using epsilon greedy
                if (random.random() < epsilon):
                    # Random action 
                    a1 = env.action_space.sample()
                    #a1 = random.randint(0, env.action_space.n-1)
                else:
                    a1 = np.argmax(Q[s,:]) 

                #Update rule
                Q[s,a] = Q[s,a] + alpha*(r + gamma*Q[s1,a1] - Q[s,a])

                a = a1
                s = s1
                G = G + r
                index = index + 1
                epsilon = epsilon * 0.99

            env.render()
            Gvector[i] = G
        return Gvector