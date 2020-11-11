# Import libs
from gym import envs
from gym.envs.toy_text import discrete, TaxiEnv
import numpy
import math
import agent as viagent

# Load and reset environment
env: TaxiEnv = envs.make('Taxi-v3')
env.reset()

# Create agent
agent = viagent.ValueIterationAgent(env,{},(5,5))
# Note: It would be possible to find all terminal states, but a bit cumbersome and the result is ok anyway.
# Note 2: The shape is not correct (doesnt for rendering), but this will have to be part of the backlog.

# Find optimal policy
agent.estimateOptimalPolicy()

# Render walking by the policy
agent.renderAppliedPolicy()

