#!/usr/bin/env python

'''
Created on August 29, 2014

@author: Jonathan Scholz <jonathan.scholz@gmail.com>
'''

import sys
import numpy as np
from agents import QLearningAgent
from domains import CliffMDP, Drawable2D


def run_episode(mdp, agent, kbd_ctl=False):
    '''

    '''

    state = mdp.get_start_state()
    step = 0
    r_total = 0
    keymap = {"w": 0, "a": 1, "s": 2, "d": 3}
    while not mdp.is_terminal(state):
        if kbd_ctl:
            print "Enter action (keymap: %s) >> " % str(keymap),
            char = sys.stdin.readline()[0]
            action = keymap[char]
        else:
            action = agent.get_action(state)
        # print "(s,a) = (%s, %s)" % (str(state), str(action))
        
        newstate = mdp.get_transition(state, action)
        reward = mdp.get_reward(newstate)
        agent.update(state, action, newstate, reward)

        if isinstance(mdp, Drawable2D):
            mdp.render(state, action_values=agent._get_action_values())

        state = newstate
        r_total += reward
        step += 1
        print "\t[%d] total reward: %f, action: %s" %\
              (step, r_total, str(action))
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
