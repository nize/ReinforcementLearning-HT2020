#from gym import envs
#env = envs.make('CliffWalking-v0')
#env.reset()
#env.render()
#env.step(env.action_space.sample())
#env.render()

#############################################################################
# This implementation is based on the algorithm of page 83 of RLbook2020.pdf
#############################################################################

# Import libs
from gym import envs
import numpy
import math

# Load and reset environment
env = envs.make('CliffWalking-v0')
env.reset()

# The environment is not correctly modelled
# The model doesn't show that no more reward will be returned after reaching the end goal
terminal_state = 47
terminal_state_P = [(1.0, 47, 0, True)],[(1.0, 47, 0, True)],[(1.0, 47, 0, True)],[(1.0, 47, 0, True)]

P = {}
for s in range(env.nS):
    if s == terminal_state:
        P[s] = terminal_state_P
    else:
        P[s] = env.P[s]


# Define function
#def conditionalProbability(P,sprime,r,s,a):
#    return 

# Configure parameters
theta = 0.1
delta = theta + 1
gamma = 0.9

# Intialize state value function
V = numpy.zeros(env.nS)
#V = random.rand(1,env.nS);
#for s in range(env.nS):
#    V[s] = random.random()*2 - 1 #todo: do better

# Estimate state value function
while delta > theta:
    delta = 0
    # work
    for s in range(env.nS):
        v = V[s]
        # Estimate V[s]
        for a in range(env.nA):
            possible_results = P[s][a]
            # Given the action, estimate the value, and find the action giving the max value
            vmax = -math.inf
            for pr in range(len(possible_results)):
                p = possible_results[pr][0]
                sprime = possible_results[pr][1]
                r = possible_results[pr][2]
                v = p*(r + gamma*V[sprime])
                if v > vmax:
                    vmax = v
            V[s] = vmax
            delta = max(delta,abs(v-V[s]))   

# Compute optimal policy based on value estimation
pi = numpy.zeros(env.nS)
for s in range(env.nS):
    vmax = -math.inf
    for a in range(env.nA):
        possible_results = P[s][a]
        for pr in range(len(possible_results)):
            p = possible_results[pr][0]
            sprime = possible_results[pr][1]
            r = possible_results[pr][2]
            v = p*(r + gamma*V[sprime])
            if v > vmax:
                vmax = v
                amax = a
    pi[s] = amax

#terminal_state = (env.shape[0] - 1, env.shape[1] - 1)
#for s in range(env.nS):
#    position = numpy.unravel_index(s, env.shape)
#    print(position)

# Apply the policy
env.reset()
inTerminal = False
while not inTerminal:
    env.render()
    [s,r,inTerminal,p] = env.step(pi[env.s])