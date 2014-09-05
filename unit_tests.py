#!/usr/bin/env python

'''
Created on August 29, 2014

@author: Jonathan Scholz <jonathan.scholz@gmail.com>
'''

import unittest
import math
from agents import QFunction, QLearningAgent, SARSAAgent
from domains import CliffMDP

class TestQFunction(unittest.TestCase):
    '''
    Tests the core Q-function used in both RL agents.
    '''

    def setUp(self):
        pass

    def test_set_and_get_item(self):
        Q = QFunction()

        Q.set((0, 0), 1, 0.1)
        self.assertEqual(Q[(0, 0), 1], 0.1)

        Q.set((0, 0), 2, 0.2)
        self.assertEqual(Q[(0, 0), 2], 0.2)

        Q.set((0, 0), 3, 0.3)
        self.assertEqual(Q[(0, 0), 3], 0.3)

    def test_slice_synchronization(self):
        Q = QFunction()
        Q[(0, 0), 3] = 0.3
        Q[(0, 0), 1] = 0.1
        Q[(0, 0), 2] = 0.2
        Q[(0, 1), 1] = -0.1
        Q[(0, 1), 2] = -0.2
        Q[(0, 1), 3] = -0.3

        self.assertEqual(Q.slice((0, 0)), {1: 0.1, 2: 0.2, 3: 0.3})
        self.assertEqual(Q.slice((0, 1)), {1: -0.1, 2: -0.2, 3: -0.3})

    def test_removal_operators(self):
        Q = QFunction()
        Q[(0, 0), 3] = 0.3
        Q[(0, 0), 1] = 0.1
        Q[(0, 0), 2] = 0.2

        Q.remove((0, 0), 1)
        self.assertEqual(Q.slice((0, 0)), {2: 0.2, 3: 0.3})

        del Q[(0, 0), 2]
        self.assertEqual(Q.slice((0, 0)), {3: 0.3})

    def test_state_set(self):
        Q = QFunction()
        Q[(0, 0), 3] = 0.3
        Q[(0, 0), 1] = 0.1
        Q[(0, 1), 1] = -0.1
        Q[(0, 1), 2] = -0.2

        self.assertEqual(Q.states, set([(0, 1), (0, 0)]))

    def test_slice_iteration(self):
        Q = QFunction()
        Q[0, 1] = 0.1
        Q[0, 2] = 0.2
        Q[1, 1] = 0.1
        Q[1, 2] = 0.2

        for state in Q.states:
            self.assertEqual(Q.slice(state), {1: 0.1, 2: 0.2})

class TestQLearningAgent(unittest.TestCase):
    '''
    Tests the Q-learning agent's action methods and backup function.
    '''

    def setUp(self):
        self.agent = QLearningAgent(legal_actions=(0, 1),
                                    gamma=0.9, alpha=0.25, epsilon=0.9)

        self.agent.Q[0, 1] = 1.0
        self.agent.Q[0, 2] = 0.5
        self.agent.Q[1, 1] = -2.0
        self.agent.Q[1, 2] = -1.0

    def test_get_greedy_action(self):
        self.assertEqual(self.agent._get_greedy_action(0), 1)
        self.assertEqual(self.agent._get_greedy_action(1), 2)

    def test_get_random_action(self):
        self.assertIn(self.agent._get_random_action(),
            self.agent.legal_actions)

    def test_td_error(self):
        self.assertTrue(abs(self.agent._td_error(0, 0, 1, 1.9) - 1.0)\
         < 1e-10)
        self.assertTrue(abs(self.agent._td_error(0, 0, 1, -0.5) + 1.4)\
         < 1e-10)

class TestSARSAAgent(unittest.TestCase):
    '''
    Tests the SARSA agent's action methods and backup function.
    '''

    def setUp(self):
        self.agent = SARSAAgent(legal_actions=(0, 1),
                                gamma=0.9, alpha=0.25, epsilon=0.9)

        self.agent.Q[0, 1] = 1.0
        self.agent.Q[0, 2] = 0.5
        self.agent.Q[1, 1] = -2.0
        self.agent.Q[1, 2] = -1.0

    def test_get_greedy_action(self):
        self.assertEqual(self.agent._get_greedy_action(0), 1)
        self.assertEqual(self.agent._get_greedy_action(1), 2)

    def test_get_random_action(self):
        self.assertIn(self.agent._get_random_action(),
            self.agent.legal_actions)

    def test_td_error(self):
        self.assertTrue(self.agent._td_error(0, 0, 1, 1.9, 2) - 1.0 < 1e-10)
        self.assertTrue(self.agent._td_error(0, 0, 1, 1.9, -2) - 1.9 < 1e-10)

class TestCliffMDP(unittest.TestCase):
    '''
    Tests the main MDP methods of the Cliff Domain.
    '''

    def setUp(self):
        self.mdp = CliffMDP(width=5, height=5)

    def test_get_start_state(self):
        self.assertEqual(self.mdp.get_start_state(), (0, 4))

    def test_get_reward(self):
        self.assertEqual(self.mdp.get_reward((0, 0)), -1.)
        self.assertEqual(self.mdp.get_reward((0, 4)), -1.)
        self.assertEqual(self.mdp.get_reward((1, 4)), -100.)
        self.assertEqual(self.mdp.get_reward((3, 4)), -100.)
        self.assertEqual(self.mdp.get_reward((4, 4)), -1.)
        self.assertEqual(self.mdp.get_reward((4, 0)), -1.)

    def test_get_transition(self):
        # test corners
        self.assertEqual(self.mdp.get_transition((0, 0), 0), (0, 0))
        self.assertEqual(self.mdp.get_transition((0, 0), 1), (0, 0))
        self.assertEqual(self.mdp.get_transition((0, 4), 1), (0, 4))
        self.assertEqual(self.mdp.get_transition((0, 4), 2), (0, 4))
        self.assertEqual(self.mdp.get_transition((4, 4), 2), (4, 4))
        self.assertEqual(self.mdp.get_transition((4, 4), 3), (4, 4))
        self.assertEqual(self.mdp.get_transition((4, 0), 0), (4, 0))
        self.assertEqual(self.mdp.get_transition((4, 0), 3), (4, 0))

        # test inside dynamics
        self.assertEqual(self.mdp.get_transition((2, 2), 0), (2, 1))
        self.assertEqual(self.mdp.get_transition((2, 2), 1), (1, 2))
        self.assertEqual(self.mdp.get_transition((2, 2), 2), (2, 3))
        self.assertEqual(self.mdp.get_transition((2, 2), 3), (3, 2))
        
    def test_is_terminal(self):
        self.assertFalse(self.mdp.is_terminal((0, 4)))
        self.assertTrue(self.mdp.is_terminal((1, 4)))
        self.assertTrue(self.mdp.is_terminal((4, 4)))
        self.assertFalse(self.mdp.is_terminal((4, 3)))
        
if __name__ == '__main__':
    unittest.main()
    