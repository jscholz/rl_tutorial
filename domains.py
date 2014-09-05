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
    values of get_transition and get_action.  The former of these
    encodes domain dynamics, while the latter encodes a possibly
    discrete set which can be either sampled from or queried with
    an index value.  Both of these types should be hashable and
    comparable (so no mutable types!)
    '''

    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self._states = None
        self._actions = None

    def get_reward(self, state):
        '''
        Return a reward for the given state.
        Note: this implementation doesn't consider rewards which depend on
        actions, because while more general it encodes model information
        in the reward function.  

        :param state: A hashable state object (e.g. a tuple)
        '''
        raise NotImplementedError()

    def get_transition(self, state, action, **kwargs):
        '''
        Returns a new state sampled from the transition distribution  
        for given state and action.  This state should be hashable.

        :param state: A hashable state object (e.g. a tuple)
        :param action: A hashable action object (e.g. an integer)
        '''
        raise NotImplementedError()
    
    def get_action(self, idx=None):
        '''
        Returns an action from some internal action representation, 
        possibly conditional on idx.  Should return random sample
        if idx is None.   

        :param idx: An optional action index.
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

        :param state: A hashable state object (e.g. a tuple)
        '''
        return False
    
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
    and colors all terminal states.  The current state is visualized
    as a blue circle, which is red in terminal states.

    This class assumes that the state and action members have been defined
    for the target MDP before initialization, and that is_terminal is 
    callable.  
    States are assumed to be (x,y) tuples corresponding to grid cells, and 
    actions are assumed to be integers in {0,1,2,3} corresponding to the 
    directions {N,W,S,E}.
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

        # main TK canvas
        self.canvas = Tkinter.Canvas(self, width=scr_width, height=scr_height,
                                     borderwidth=0, highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand="true")

        # containers and helpers for canvas items
        self._rect = {}      # visualize grid cells
        self._oval = {}      # visualize current state
        self._wedge = {}     # visualize action values
        self._vmax = -1       # for color scaling actions
        self._vmin = -2       # for color scaling actions

        # color definitions
        self._bg_color = "light grey"
        self._sprite_color = "dark blue"
        self._terminal_color = "red"

        # iterate through the defined states
        for state in self.states:
            x0 = state[0] * self.cellwidth
            y0 = state[1] * self.cellheight
            x1 = x0 + self.cellwidth
            y1 = y0 + self.cellheight

            # draw containers for grid cells
            cflag = "terminal" if self.is_terminal(state) else "default"
            self._rect[state] = self.canvas.create_rectangle(
                x0, y0, x1, y1, fill=self._bg_color, tags=("rect", cflag))
            
            # draw containers for actions
            for action in self.actions:
                self._wedge[(state, action)] = self.canvas.create_arc(
                    x0, y0, x1, y1,
                    start=action * 90 + 45,
                    extent=90, fill=self._bg_color, tags="qvals")

            # draw containers for sprite state
            self._oval[state] = self.canvas.create_oval(
                x0+5, y0+5, x1-5, y1-5, fill=self._bg_color, tags="oval")

        # draw to screen (necessary if not calling TK mainloop)
        self.update()
    
    def render(self, state, action_values=None):
        '''
        Redraws the buffer to visualize the current state.

        Also implements a method for visualizing policies using the 
        action values, if provided.
        For grid domains, assumes actions are the standard NWSE, and 
        represented in that order using the integers 0:3.  

        :param state: The state to render
        :param action_values: A dictionary of (s,a) tuples and their 
                              corresponding values (optional)
        '''
        if not hasattr(self, 'canvas') or not self._oval.has_key(state):
            return

        # fill all the keyed cells by color
        self.canvas.itemconfig("rect", fill=self._bg_color)
        self.canvas.itemconfig("terminal", fill=self._terminal_color)
        self.canvas.itemconfig("oval", fill=self._bg_color)

        # color the target state
        item_id = self._oval[state]
        color = self._terminal_color if self.is_terminal(state)\
                                     else self._sprite_color
        self.canvas.itemconfig(item_id, fill=color)

        # draw policy if provided
        if action_values:
            for state in action_values.states:
                qvals = action_values.slice(state)
                _, vmax = max(qvals.iteritems(), key=lambda x: x[1])
                _, vmin = min(qvals.iteritems(), key=lambda x: x[1])
                for action, value in qvals.iteritems():
                    if self._wedge.has_key((state, action)):
                        try:
                            color_val = (value - vmin) / (vmax - vmin)
                        except:
                            color_val = 0.5
                        if np.isnan(color_val):
                            color_val = 1.0
                        if color_val < 0.5:
                            # scaled red for low-value actions
                            color = '#%02x%02x%02x' % (color_val * 255, 0, 0)
                        else:
                            # scaled green for high-value actions
                            color = '#%02x%02x%02x' % (0, color_val * 255, 0)

                        item_id = self._wedge[(state, action)]
                        self.canvas.itemconfig(item_id, fill=color)

        # draw (no after calls since mainloop not running)
        self.canvas.update_idletasks()

class CliffMDP(MDP, GridVisualizer):
    '''
    Implements the cliff-world problem as described by Sutton & Barto:
    http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node65.html
    '''

    def __init__(self, width=12, height=4, render=False, *args, **kwargs):
        '''
        Construct a Cliff-MDP with the provided dimensions.

        :param width: The width, in grid cells
        :param height: The height, in grid cells
        '''
        MDP.__init__(self, *args, **kwargs)
        self.width = width
        self.height = height

        # define range of cliff states along bottom border
        self.cliff_idxs = np.array([[1, height-1], [width-2, height-1]])

        # define state space corresponding to cells on the grid
        self._states = [(i, j) for i in xrange(width) 
            for j in xrange(height)]
        self._states_gen = ((i, j) for i in xrange(width) 
            for j in xrange(height))

        # define actions corresponding to the cardinal directions {N,W,S,E}
        self._actions = [(i) for i in xrange(4)]
        # define action effects in screen coords
        self._action_effects = [(0, -1), (-1, 0), (0, 1), (1, 0)]

        if render:
            # initialize visualizer last, b/c depends on MDP is_terminal method
            GridVisualizer.__init__(self)

    def get_reward(self, state):
        '''
        Return a reward for the given state.

        :param state: The query state.
        '''
        if self._is_cliff(state):
            return -100.
        else:
            return -1.
    
    def get_transition(self, state, action, **kwargs):
        '''
        Returns a new state sampled from the transition distribution 
        for given state and action.  This state should be hashable.

        :param state: The starting state
        :param action: The action to execute
        '''

        ds = self._action_effects[action]
        newstate = (np.clip(state[0] + ds[0], 0, self.width - 1),
                    np.clip(state[1] + ds[1], 0, self.height - 1))
        return newstate
    
    def get_action(self, idx=None):
        '''
        Returns an action from some internal action representation, 
        possibly conditional on idx.  Should return random sample
        if idx is None.   

        :param idx: An action index.  If None a random action is returned
        '''
        if idx is None:
            idx = np.random.choice(self.actions)[0]
        return self.actions[idx]
    
    def get_start_state(self):
        '''
        Returns the starting state for the MDP.  Should be hashable
        '''
        return (0, self.height - 1)
    
    def is_terminal(self, state):
        '''
        Returns true if given state is terminal.

        :param state: A state tuple
        '''
        return self._is_cliff(state) or (state == (self.width - 1, 
                                                   self.height - 1))

    def _is_cliff(self, state):
        '''
        Tests whether the given state is on the cliff -- a range 
        of terminal states with large negative reward.  

        :param state: A state tuple
        '''
        if state[0] >= self.cliff_idxs[0][0] and\
           state[0] <= self.cliff_idxs[1][0] and\
           state[1] >= self.cliff_idxs[0][1] and\
           state[1] <= self.cliff_idxs[1][1]:
            return True
        else:
            return False
    
    def print_state(self, state):
        '''
        Visualizes the cliff domain state in the terminal.
        '''
        for j in xrange(self.height):
            for i in xrange(self.width):
                if (i, j) == state:
                    print "*",
                elif self._is_cliff((i, j)):
                    print "_",
                else:
                    print u"\u25A1",
            print ""
        
        print "\033[%dA" % (self.height+1)

if __name__ == "__main__":  

    cmdp = CliffMDP(width=5, height=5, render=True)

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
