#############################################################################
# This implementation is based on the algorithm of page 83 of RLbook2020.pdf
#############################################################################

# Import libs
from gym import envs
from gym.envs.toy_text import discrete
import numpy
import math
import functools
import operator

class ValueIterationAgent:
    def __init__(self,env: discrete.DiscreteEnv):
        self.env: discrete.DiscreteEnv = env
        self.env.reset()
        # Intialize state value function
        self.V  = numpy.zeros(self.env.nS)
        # Configure parameters (todo: pass as optional args)
        self.theta: float = 0.1
        self.gamma: float = 0.99

    def estimateOptimalPolicy(self):
        """
        docstring
        """

        delta: float = self.theta + 1
        #V = random.rand(1,env.nS);
        #for s in range(env.nS):
        #    V[s] = random.random()*2 - 1

        # Estimate state value function
        while delta > self.theta:
            delta = 0
            for s in range(self.env.nS):
                # Estimate V[s]
                v = self.V[s]
                # Find action giving the greatest value
                self.V[s] = max([self.estimateStateActionValue(s,a) for a in range(self.env.nA)])
                delta = max(delta,abs(v-self.V[s]))

        # Compute optimal policy based on value estimation
        self.pi = numpy.zeros(self.env.nS,dtype=int)
        for s in range(self.env.nS):
            amax = numpy.argmax([self.estimateStateActionValue(s,a) for a in range(self.env.nA)])
            self.pi[s] = int(amax)


    def estimateStateActionValue(self,s: int,a: int) -> float:
        possible_outcomes = self.env.P[s][a]
        v = functools.reduce(operator.add,map(self.outcomeValueContribution,possible_outcomes))
        return v

    def outcomeValueContribution(self,outcome) -> float:
        p = outcome[0]
        sprime = outcome[1]
        r = outcome[2]
        v = p*(r + self.gamma*self.V[sprime])
        return v

    def renderPolicy(self):
        print(self.pi.reshape(5,5))
    
    def renderStateValue(self):
        print(self.V.reshape(5,5)) #print(self.V.reshape(self.env.shape))

    def renderAppliedPolicy(self) -> float:
        self.env.reset()
        inTerminal = False
        totr = 0
        while not inTerminal:
            self.env.render()
            [s,r,inTerminal,p] = self.env.step(self.pi[self.env.s])
            totr=totr+r
        return totr
