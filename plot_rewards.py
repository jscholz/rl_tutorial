#!/usr/bin/env python

'''
Created on September 2, 2014

@author: Jonathan Scholz <jonathan.scholz@gmail.com>
'''

import numpy as np
from matplotlib import pyplot as plt
import pickle

if __name__ == '__main__':
    
    sarsa_episode_rewards = pickle.load(open('sarsa_episode_rewards.pkl','rb'))
    qlearning_episode_rewards = pickle.load(open('qlearning_episode_rewards.pkl','rb'))


    plt.clf()
    plt.ylabel('Reward per episodes')
    plt.xlabel('Episodes')
    # plt.title('')
    plt.plot(sarsa_episode_rewards)
    plt.plot(qlearning_episode_rewards)
    # plt.ylim((min(-50, min(mcmc_logp)),0))
    # plt.xlim((0, max(30,len(mcmc_logp))))
    plt.show()