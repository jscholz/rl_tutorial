#!/usr/bin/env python

'''
Created on August 28, 2014

@author: Jonathan Scholz <jonathan.scholz@gmail.com>
'''


import numpy as np
import Tkinter
import time

class MDP(object):
    '''
    A base class for markov decision problems, which must define at
    a minimum: a reward function, an action set, a transition model,
    and starting and terminal distributions.
    
    The state and action space is determined implicitly by the return
    values of getTransition and getAction.  The former of these
    encodes domain dynamics, while the latter encodes a possibly 
    discrete set which can be either sampled from or queried with 
    an index value.  Both of these types should be hashable and 
    comparable (so no mutable types!*)
    
    * so if lists and or float is necessary, consider wrapping it 
    with MDPElement   
    '''
    
    def __init__(self):
        self._states = None
        self._actions = None
        self.gamma = None        
    
    def get_reward(self, state, action):
        '''
        Return a reward for the given state and action.
            * Don't forget: if this function steps the world for the given action
              before computing a reward then it encodes some model information,
              which may be relevant for model-learning experiments
        :param state:
        :param action:
        '''
        raise NotImplementedError()
    
    def get_transition(self, state, action, **kwargs):
        '''
        Returns a new state sampled from the transition distribution  
        for given state and action.  This state should be hashable.
        '''
        raise NotImplementedError()
    
    def get_action(self, idx = None):
        '''
        Returns an action from some internal action representation, 
        possibly conditional on idx.  Should return random sample
        if idx is None.   
        '''
        raise NotImplementedError()
    
    def get_start_state(self):
        '''
        Returns the starting state for the MDP.  Should be hashable
        '''
        raise NotImplementedError()
    
    def is_terminal(self, state):
        '''
        Returns true if given state is terminal 
        '''
        return False
    
    @classmethod
    def getDistance(cls, s1, s2):
        '''
        Implements a state-space distance function 
        :param cls:
        :param s1: First state
        :param s2: Second state
        :return tuple containing scalar distance and an error vector
        '''
        raise NotImplementedError()
    
    # define read-only properties for states and actions
    # Note: not all MDP's will define these explicitly
    states = property(lambda self: self._states)
    n_states = property(lambda self: len(self._states))
    
    actions = property(lambda self: self._actions)
    n_actions = property(lambda self: len(self._actions))

class Drawable2D(object):
    '''
    An abstract mixin class for visualizing 2D domains.  
    '''
    def render(self, state):
        '''
        Renders the provided state.
        '''
        raise NotImplementedError()

class GridVisualizer(Tkinter.Tk, Drawable2D):
    '''
    A generic mixin class which defines a draw method for 2D grid-based MDPs.  

    Without any customization, draws a blue grid of the appropriate size, 
    and colors all terminal states yellow.  The current state is visualized
    as a green circle, which is red in terminal states.

    This class assumes that width and height members are defined before
    initialization, and that is_terminal is callable.  
    '''
    def __init__(self, *args, **kwargs):
        '''

        Assumes height and width members are defined.
        '''
        Tkinter.Tk.__init__(self, *args, **kwargs)
        self.cellwidth = 50
        self.cellheight = 50
        scr_width = self.width * self.cellwidth
        scr_height = self.height * self.cellheight

        self.canvas = Tkinter.Canvas(self, width=scr_width, height=scr_height,
                                     borderwidth=0, highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand="true")

        self.rect = {}
        self.oval = {}
        for column in xrange(self.width):
            for row in xrange(self.height):
                x1 = column * self.cellwidth
                y1 = row * self.cellheight
                x2 = x1 + self.cellwidth
                y2 = y1 + self.cellheight
                state = (column, row)
                flag = "terminal" if self.is_terminal(state) else "nonterminal"
                self.rect[state] = self.canvas.create_rectangle(
                    x1,y1,x2,y2, fill="blue", tags=("rect", flag))
                self.oval[state] = self.canvas.create_oval(
                    x1+2,y1+2,x2-2,y2-2, fill="blue", tags="oval")

        self.update()
        

    def render(self, state):
        self.canvas.itemconfig("rect", fill="blue")
        self.canvas.itemconfig("terminal", fill="yellow")
        self.canvas.itemconfig("oval", fill="blue")
        if not self.oval.has_key(state):
            return
        item_id = self.oval[state]
        color = "red" if self.is_terminal(state) else "green"
        self.canvas.itemconfig(item_id, fill=color)
        self.canvas.update_idletasks()

class CliffMDP(MDP, GridVisualizer):
    '''
    Implements the cliff-world problem as described by Sutton & Barto:
    http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node65.html
    '''

    def __init__(self, width=12, height=4):
        '''
        Construct a Cliff-MDP with the provided dimensions.
        '''
        self.width = width
        self.height = height
        MDP.__init__(self)

        # define range of cliff states
        self.cliff_idxs = np.array([[1, height-1], [width-2, height-1]])

        # define state space corresponding to cells on the grid
        self._states = [(i, j) for i in xrange(width) for j in xrange(height)]
        self._states_gen = ((i, j) for i in xrange(width) for j in xrange(height))

        # define actions corresponding to the cardinal directions {N,E,S,W}
        self._actions = [(i) for i in xrange(4)]
        self._action_effects = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # initialize visualizer last, as it depends on MDP is_terminal method
        GridVisualizer.__init__(self)

    def get_reward(self, state, action):
        '''
        Return a reward for the given state and action.
            * Don't forget: if this function steps the world for the given action
              before computing a reward then it encodes some model information,
              which may be relevant for model-learning experiments
        :param state:
        :param action:
        '''
        if self.is_cliff(state):
            return -100.
        else:
            return -1.
    
    def get_transition(self, state, action, **kwargs):
        '''
        Returns a new state sampled from the transition distribution  
        for given state and action.  This state should be hashable.
        '''
        ds = self._action_effects[action]
        newstate = (np.clip(state[0] + ds[0], 0, self.width),
                    np.clip(state[1] + ds[1], 0, self.height))
        return newstate
    
    def get_action(self, idx = None):
        '''
        Returns an action from some internal action representation, 
        possibly conditional on idx.  Should return random sample
        if idx is None.   
        '''
        if idx is None:
            idx = np.random.choice(self.actions)[0]
        return self.actions[idx]
    
    def get_start_state(self):
        '''
        Returns the starting state for the MDP.  Should be hashable
        '''
        return (0, self.height-1)
    
    def is_terminal(self, state):
        '''
        Returns true if given state is terminal.
        '''
        return self.is_cliff(state) or (state == (self.width, 0))

    def is_cliff(self, state):
        '''
        Tests whether the given state is on the cliff -- a range 
        of terminal states with large negative reward.  
        '''
        if state[0] >= self.cliff_idxs[0][0] and state[0] <= self.cliff_idxs[1][0] and\
           state[1] >= self.cliff_idxs[0][1] and state[1] <= self.cliff_idxs[1][1]:
            return True
        else:
            return False
    
    def print_state(self, state):
        for j in xrange(self.height):
            for i in xrange(self.width):
                if (i, j) == state:
                    print "*",
                elif self.is_cliff((i, j)):
                    print "_",
                else:
                    print u"\u25A1",
            print ""
        
        print "\033[%dA" % (self.height+1)

if __name__ == "__main__":  

    cmdp = CliffMDP(width=12, height=4)

    state = cmdp.get_start_state()
    cmdp.render(state)

    while not cmdp.is_terminal(state):
        action = cmdp.get_action() # get a random action
        state = cmdp.get_transition(state, action)
        cmdp.print_state(state)
        cmdp.render(state)
        if cmdp.is_terminal(state):
            state = cmdp.get_start_state()
        time.sleep(0.1)
