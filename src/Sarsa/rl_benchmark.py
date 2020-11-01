import matplotlib.pyplot as plt
import sarsa


episodes = 3000
steps = 200
evnironment = 'Taxi-v3' #'HotterColder-v0' #'FrozenLake-v0'

Sarsa = sarsa.Sarsa()

Gvector = Sarsa.run(episodes, 
                    steps, 
                    evnironment)

plt.plot(Gvector)
plt.show()



