#!/usr/bin/env python

'''
Created on August 29, 2014

@author: Jonathan Scholz <jonathan.scholz@gmail.com>
'''


import numpy as np
from collections import defaultdict, deque

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
        '''
        Returns an action for the provided state.  Use this function to
        implement the exploration policy for the agent. 
        '''
        raise NotImplementedError()

    def update(self, state, action, newstate, reward):
        '''
        Implements the update rule for the learning agent.  Called every time
        the agent executes an action in the target domain.

        :param state: The starting state for the backup
        :param action: The action executed in that state
        :param newstate: The resulting state
        :param reward: The reward obtained
        '''
        self.r_total += reward

class QFunction(object):
    '''
    Implements a dictionary-based Q-function for reinforcemnt learning 
    applications.  In addition to a pass-through interface to the underlying
    dictionary keyed by (state, action) tuples, QFunction also supports
    constant-time slicing by state for the purpose of max/argmax. 

    E.G. 
    >>> Q[(0,1), 1] = 0.1
    >>> Q[(0,1), 2] = 0.2
    >>> Q[(0,1), 3] = 0.3
    >>> print Q.slice((0,1))
    ... {1: 0.1, 2: 0.2, 3: 0.3}

    Slicing is implemented using a second internal dictionary which keeps 
    track of the actions defined for each state entry in the Q table.  
    
    For convenience, QFunction defines a sequence interface to the main Q 
    table.  These functions call the provided interface methods, which assume
    (state, action) arguments.  It is possible to break these checks by 
    passing in a single state of length 2, but fixing this with official 
    state and action types wouldn't be very pythonic.
    '''

    def __init__(self):
        self._Q = defaultdict(float) # Q-function; default to zero
        self._V = defaultdict(set)   # Value-function; default to empty set
        self._S = set()              # State set: keeps track of defined states

    def __str__(self):
        return str(self._Q)

    def __repr__(self):
        return str(self._Q)

    def __getitem__(self, key):
        return self._Q[key]

    def __setitem__(self, key, value):
        assert(len(key) == 2)
        self.set(key[0], key[1], value)

    def __delitem__(self, key):
        assert(len(key) == 2)
        self.remove(key[0], key[1])

    def __iter__(self):
        return iter(self._Q)

    def get(self, state, action):
        '''
        Retrieves the specified (state, action) entry.

        :param state: The query state
        :param action: The query action
        '''
        return self._Q[state, action]

    def set(self, state, action, value):
        '''
        Adds the given state-action value to the Q-function, and updates
        internal structures as necessary.

        :param state: The query state
        :param action: The query action
        :param value: The value to assign
        '''
        self._Q[state, action] = value
        self._V[state].add(action)
        self._S.add(state)

    def remove(self, state, action):
        '''
        Removes the given state-action pair from the Q-function, and updates
        internal structures as necessary.

        :param state: The query state
        :param action: The query action
        '''
        self._Q.pop((state, action))
        self._V[state].remove(action)

        if len(self._V[state]) == 0:
            self._S.remove(state)

    def slice(self, state):
        '''
        Returns a slice of the Q-function for the provided state as a
        dictionary, keyed by action. 

        :param state: The query state
        '''
        return {action: self._Q[state, action] for action in self._V[state]}

    # read-only access to state set
    states = property(lambda self: self._S)

class QFunctionAgent(RLAgent):
    '''
    Implements a basic QFunction-based learning agent, using an epsilon-greedy
    exploration policy.  
    '''

    def __init__(self, alpha=0.5, epsilon=0.9, *args, **kwargs):
        super(QFunctionAgent, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = QFunction()

    def _get_action_values(self, state=None):
        '''
        Returns the dictionary of q-values for the current state.
        If no state is provided, returns the entire Q table.

        :param state: The query state (optional)
        '''
        if state is not None:
            return self.Q.slice(state)
        else:
            return self.Q

    def _get_greedy_action(self, state):
        '''
        Returns the greedy policy for the provided state, which maximizes
        the expected return given the agent's current Q-function.

        :param state: The query state
        '''
        Q_s = self._get_action_values(state)

        if len(Q_s) == 0:
            # if no values exist yet, initialize a legal action to zero
            Q_s = {self._get_random_action(): 0.}

        # select the argmax action
        key, value = max(Q_s.iteritems(), key=lambda x: x[1])

        return key

    def _get_random_action(self, state=None):
        '''
        Returns a random action that is legal in the current state.

        :param state: The query state 
        '''
        idx = np.random.choice(self.legal_actions)
        return self.legal_actions[idx]

    def get_action(self, state):
        '''
        Implements an epsilon-greedy exploration policy.

        :param state: The query state
        '''
        if np.random.rand() > self.epsilon:
            return self._get_greedy_action(state)
        else:
            return self._get_random_action(state)
    
    def _td_error(self, state, action, newstate, reward):
        raise NotImplementedError()

    def update(self, state, action, newstate, reward):
        '''
        Implements a Q-learning (off-policy) backup operation.
        '''
        super(QFunctionAgent, self).update(state, action, newstate, reward)

        self.Q[state, action] = self.Q[state, action] + self.alpha * \
            self._td_error(state, action, newstate, reward)

class QLearningAgent(QFunctionAgent):
    '''
    Implements a Q-Learning agent.
    '''

    def _td_error(self, state, action, newstate, reward):
        '''
        Computes the temporal-difference error for a Q-learning (off-policy)
        backup operation.
        '''
        Q_s = self.Q.slice(newstate).values()
        if len(Q_s) == 0:
            Q_s = [0.]

        return reward + self.gamma * np.max(Q_s) - self.Q[state, action]

class SARSAAgent(QFunctionAgent):
    '''
    Implements a SARSA agent.  
    '''
    
    def __init__(self, *args, **kwargs):
        super(SARSAAgent, self).__init__(*args, **kwargs)
        self._next_action = None
        self._next_state = None

    def get_action(self, state):
        '''
        For SARSA we must select an action for the next state during update 
        calls.  For this to be on-policy, we of course need to actually execute
        that action, but only if the query state hasn't changed (i.e. if the
        episode ended).  
        '''
        if state == self._next_state and self._next_action is not None:
            return self._next_action
        else:
            return super(SARSAAgent, self).get_action(state)

    def _td_error(self, state, action, newstate, reward, next_action=None):
        '''
        Computes the temporal-difference error for a SARSA (on-policy)
        backup operation.
        '''
        if next_action is None:
            self._next_state = None # forces e-greedy even if state==newstate
            self._next_action = self.get_action(newstate)
        else:
            self._next_action = next_action

        self._next_state = newstate

        Q_sa_next = self.Q[self._next_state, self._next_action]

        return reward + self.gamma * Q_sa_next - self.Q[state, action]

if __name__ == '__main__':
    Q = QFunction()

    # test set method
    Q.set((0,0), 3, 0.3)
    Q.set((0,0), 1, 0.1)
    Q.set((0,0), 2, 0.2)

    # test getitem
    assert(Q[(0,0), 1] == 0.1)
    assert(Q[(0,0), 2] == 0.2)
    assert(Q[(0,0), 3] == 0.3)

    # test setitem
    Q[(0,1), 1] = -0.1
    Q[(0,1), 2] = -0.2
    Q[(0,1), 3] = -0.3

    # test slice synchronization 
    print Q.slice((0,0))
    assert(Q.slice((0,0)) == {1: 0.1, 2: 0.2, 3: 0.3})
    print Q.slice((0,1))
    assert(Q.slice((0,1)) == {1: -0.1, 2: -0.2, 3: -0.3})

    # test removal operators
    Q.remove((0,0), 1)
    assert(Q.slice((0,0)) == {2: 0.2, 3: 0.3})
    del Q[(0,0), 2]
    assert(Q.slice((0,0)) == {3: 0.3})

    # test state set
    print Q.states
    assert(Q.states == set([(0, 1), (0, 0)]))

    print Q
    import ipdb;ipdb.set_trace()