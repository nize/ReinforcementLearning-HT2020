# Import libs
from gym import envs
from gym.envs.toy_text import discrete
import numpy
import math
import agent_cliff as viagent

# Load and reset environment
env: discrete.DiscreteEnv = envs.make('CliffWalking-v0')
env.reset()

# Create agent
agent = viagent.ValueIterationAgent(env)

# Find optimal policy
agent.estimateOptimalPolicy()

# Render walking by the policy
agent.renderAppliedPolicy()
