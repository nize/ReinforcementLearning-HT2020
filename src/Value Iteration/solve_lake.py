# Import libs
from gym import envs
from gym.envs.toy_text import discrete
import numpy
import math
import agent as viagent

# Load and reset environment
env: discrete.DiscreteEnv = envs.make('FrozenLake-v0')
env.reset()

# Create agent
agent = viagent.ValueIterationAgent(env,{15},(4,4))

# Find optimal policy
agent.estimateOptimalPolicy()

# Render walking by the policy
agent.renderAppliedPolicy()

agent.renderPolicy()
agent.renderStateValue()
