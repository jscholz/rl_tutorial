#!/usr/bin/env python

'''
Created on August 29, 2014

@author: Jonathan Scholz <jonathan.scholz@gmail.com>
'''


import numpy as np

class RLAgent(object):
    '''
    
    '''
    def __init__(self, legal_actions, gamma, *args, **kwargs):
        '''
        Initialize the RL agent with the required domain parameters.

        :param gamma: MDP discount factor
        :param legal_actions: A list of legal actions
        '''
        super(RLAgent, self).__init__()
        self.legal_actions = legal_actions
        self.gamma = gamma

        # accumulated reward
        self.r_total = 0
        
    def get_action(self, state):
        '''Returns an action for the provided state.  Use this function to
        implement the exploration policy for the agent.  
        '''
        raise NotImplementedError()

    def update(self, state, action, newstate, reward):
        '''
        Implements the update rule for the learning agent.  Called every time
        the agent executes an action in the target domain.
        '''
        self.r_total += reward

class QLearningAgent(RLAgent):
    '''
    Implements a basic Q-learning agent.
    '''
    class QFunction(dict):
        def slice(self, el, pos = 0):
            '''
            Slices the Q-function by looking for the the provided 
            element by checking the key tuples at the specified position
            (state is pos=0, action is pos=1).  Way less efficient 
            than a numpy slices, but this is Q-learning...
            '''
            return {k:v for k, v in self.iteritems() if k[pos] == el}
    
    def __init__(self, alpha=0.5, epsilon=0.9, *args, **kwargs):
        super(QLearningAgent, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = QLearningAgent.QFunction()

    def _get_greedy_action(self, state):
        '''
        Returns the greedy policy for the provided state, which maximizes
        the expected return given the agent's current Q-function.

        :param state: The query state 
        '''
        Q_s = self.Q.slice(state)
        if len(Q_s) == 0:
            # if no values exist yet, initialize a legal action to zero
            Q_s = {(state, self._get_random_action()): 0.}

        # select the argmax action
        key, value = max(Q_s.iteritems(), key=lambda x:x[1])
        return key[1]

    def _get_random_action(self, state=None):
        '''
        Returns a random action that is legal in the current state.

        :param state: The query state 
        '''
        idx = np.random.choice(self.legal_actions)[0]
        return self.legal_actions[idx]

    def get_action(self, state):
        '''
        Implements an epsilon-greedy exploration policy.
        '''
        if np.random.rand() > self.epsilon:
            return self._get_greedy_action(state)
        else:
            return self._get_random_action(state)
            
    def update(self, state, action, newstate, reward):
        super(QLearningAgent, self).update(state, action, newstate, reward)
        s = state
        a = action
        #self.Q[s,a] = self.Q[s,a] + self.alpha * (reward + self.gamma * np.max(self.Q[newstate,:]) - self.Q[s,a])
        if not self.Q.has_key((s, a)):
            Q_sa = 0.
        else:
            Q_sa = self.Q[(s, a)]
        Q_s = self.Q.slice(newstate).values()
        if len(Q_s) == 0:
            Q_s = [0.]
        self.Q[(s, a)] = Q_sa + self.alpha * (reward + self.gamma * np.max(Q_s) - Q_sa)

if __name__ == '__main__':
    pass
