

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