#!/usr/bin/env python

'''
Created on August 29, 2014

@author: Jonathan Scholz <jonathan.scholz@gmail.com>
'''

import sys
import numpy as np
from matplotlib import pyplot as plt
import pickle

from agents import QLearningAgent, SARSAAgent
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

def train_agent(mdp, agent, max_episodes, epsilon_decay=0.9, plot=False):
    '''
    '''
    episode_rewards = []
    for i in range(max_episodes):
        episode_rewards.append(run_episode(mdp, agent, kbd_ctl=False))
        agent.epsilon *= epsilon_decay

        if plot:
            plt.interactive(True)
            plt.clf()
            plt.ylabel('Reward per episodes')
            plt.xlabel('Episodes')
            # plt.title('')
            plt.plot(episode_rewards)
            # plt.ylim((min(-50, min(mcmc_logp)),0))
            # plt.xlim((0, max(30,len(mcmc_logp))))
            plt.draw()

        print "[episode %d] episode reward: %f.  Epsilon now: %f" %\
            (i, episode_rewards[-1], agent.epsilon)

    return episode_rewards

if __name__ == '__main__':
    mdp = CliffMDP(12, 4)
    
    Agent = QLearningAgent
    # Agent = SARSAAgent
    
    agent = Agent(legal_actions=mdp.actions, gamma=mdp.gamma,
                           alpha=0.25, epsilon=0.9)

    episode_rewards = train_agent(mdp, agent, max_episodes=500, 
        epsilon_decay=0.995, plot=True)

    f = open('episode_rewards.pkl', 'wb')
    pickle.dump(episode_rewards, f)
    f.close()

    import ipdb;ipdb.set_trace()