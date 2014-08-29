#!/usr/bin/env python

'''
Created on August 29, 2014

@author: Jonathan Scholz <jonathan.scholz@gmail.com>
'''

import numpy as np
from agents import QLearningAgent
from domains import CliffMDP, Drawable2D

def run_episode(mdp, agent):
    '''

    '''

    state = mdp.get_start_state()
    step = 0
    r_total = 0
    while not mdp.is_terminal(state):
        action = agent.get_action(state)
        newstate = mdp.get_transition(state, action)
        reward = mdp.get_reward(newstate)
        agent.update(state, action, newstate, reward)

        if isinstance(mdp, Drawable2D):
        	mdp.render(state)

        state = newstate
        r_total += reward
        step += 1
        print "\t[%d] total reward: %f, action: %s" % (step, r_total, str(action))
    return r_total

def train_agent(mdp, agent, max_episodes, epsilon_decay=0.9):
	'''
	'''
	r_total = 0
	for i in range(max_episodes):
		r_total += run_episode(mdp, agent)
		agent.epsilon *= epsilon_decay
		print "[episode %d] total reward: %f" % (i, r_total)

if __name__ == '__main__':
    mdp = CliffMDP(12, 4)
    agent = QLearningAgent(legal_actions=mdp.actions, gamma=mdp.gamma,
    					   alpha=0.5, epsilon=0.9)
    # run_episode(mdp, agent)
    train_agent(mdp, agent, 100, 0.9)
