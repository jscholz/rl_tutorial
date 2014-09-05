#!/usr/bin/env python

'''
Created on September 2, 2014

@author: Jonathan Scholz <jonathan.scholz@gmail.com>
'''

from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
import pickle

def plot_sarsa_vs_qlearning(sarsa_rewards, qlearning_rewards):
    '''
    Generates a smoothed plot of sarsa and q-learning rewards,
    using scipy's UnivariateSpline.
    '''
    # plt.interactive(True)
    plt.figure(0)
    plt.clf()
    plt.ylabel('Reward per episodes')
    plt.xlabel('Episodes')

    smooth_factor = 150000

    x = range(len(sarsa_rewards))
    sms = UnivariateSpline(x, sarsa_rewards, s=245000)
    # plt.plot(x, sarsa_rewards)
    plt.plot(x, sms(x))

    smq = UnivariateSpline(x, qlearning_rewards, s=300000)
    # plt.plot(x, qlearning_rewards)
    plt.plot(x, smq(x))
    
    plt.legend(["SARSA", "Q-Learning"], loc=0)
    plt.show()

if __name__ == '__main__':

    sarsa_episode_rewards = pickle.load(open('SARSAAgent_rewards_alpha-0.25_gamma-0.9_epsilon-0.99_epsilon_decay-0.99_plot-True_max_episodes-500.pkl','rb'))
    qlearning_episode_rewards = pickle.load(open('QLearningAgent_rewards_alpha-0.25_gamma-0.9_epsilon-0.99_epsilon_decay-0.99_plot-True_max_episodes-500.pkl','rb'))
    plot_sarsa_vs_qlearning(sarsa_episode_rewards, qlearning_episode_rewards)