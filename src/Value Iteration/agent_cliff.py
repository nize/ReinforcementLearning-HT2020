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
        # add check if it cliff env
        self.fixCliffEnvironment()
        # Intialize state value function
        self.V  = numpy.zeros(self.env.nS)
        # Configure parameters (todo: pass as optional args)
        self.theta: float = 0.5
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
        self.pi = numpy.zeros(self.env.nS)
        for s in range(self.env.nS):
            amax = numpy.argmax([self.estimateStateActionValue(s,a) for a in range(self.env.nA)])
            self.pi[s] = amax


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


    def renderAppliedPolicy(self) -> float:
        self.env.reset()
        inTerminal = False
        totr = 0
        while not inTerminal:
            self.env.render()
            [s,r,inTerminal,p] = self.env.step(self.pi[self.env.s])
            totr=totr+r
        return totr

    def fixCliffEnvironment(self):
        # The environment is not correctly modelled
        # The model doesn't show that no more reward will be returned after reaching the end goal
        terminal_state = 47
        terminal_state_P = [(1.0, 47, 0, True)],[(1.0, 47, 0, True)],[(1.0, 47, 0, True)],[(1.0, 47, 0, True)]
        #self.env.P = {}
        for s in range(self.env.nS):
            if s == terminal_state:
                self.env.P[s] = terminal_state_P
