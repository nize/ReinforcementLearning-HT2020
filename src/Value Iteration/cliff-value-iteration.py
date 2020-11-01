#from gym import envs
#env = envs.make('CliffWalking-v0')
#env.reset()
#env.render()
#env.step(env.action_space.sample())
#env.render()

from gym import envs
env = envs.make('CliffWalking-v0')
env.reset()
theta = 0.1
delta = 0
while delta < theta:
    # work
    pass