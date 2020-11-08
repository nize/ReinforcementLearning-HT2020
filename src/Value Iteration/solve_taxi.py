# Import libs
from gym import envs
from gym.envs.toy_text import discrete
import numpy
import math
import agent_taxi as viagent

# Load and reset environment
env: discrete.DiscreteEnv = envs.make('Taxi-v3')
env.reset()

# Create agent
agent = viagent.ValueIterationAgent(env)

# Find optimal policy
agent.estimateOptimalPolicy()

# Render walking by the policy
agent.renderAppliedPolicy()
