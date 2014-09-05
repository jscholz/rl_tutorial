#!/usr/bin/env python

'''
Created on August 29, 2014

@author: Jonathan Scholz <jonathan.scholz@gmail.com>
'''

import sys
from matplotlib import pyplot as plt
import pickle
import argparse


from agents import QLearningAgent, SARSAAgent
from domains import CliffMDP, Drawable2D
from plot_rewards import plot_sarsa_vs_qlearning

def run_episode(mdp, agent, kbd_ctl=False, verbose=False, ascii_vis=False):
    '''
    Runs a single episode of the RL experiment.

    :param mdp: The mdp which implements the domain
    :param agent: The RL agent to train
    :param kbd_ctl: If true, use keyboard rather than agent control 
    :param verbose: If true, print 1-step rewards
    :param ascii_vis: If true, print ascii visualization
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

        if ascii_vis and hasattr(mdp, 'print_state'):
            mdp.print_state(state)

        state = newstate
        r_total += reward
        step += 1
        if verbose:
            print "\t[%d] total reward: %f, action: %s" %\
                  (step, r_total, str(action))
    return r_total

def train_agent(mdp, agent, max_episodes, epsilon_decay=0.9, plot=False):
    '''
    Trains an agent on the given MDP for the specified number of episodes.

    :param mdp: The mdp which implements the domain
    :param agent: The RL agent to train
    :param max_episodes: The maximum number of episodes to run
    :param epsilon_decay: The per-episode decay rate of the epsilon parameter
    :param plot: If true, plot the reward results online.
    '''
    episode_rewards = []
    for i in range(max_episodes):
        episode_rewards.append(run_episode(mdp, agent, kbd_ctl=False))
        if i % 1 == 0:
            agent.epsilon *= epsilon_decay

        if plot:
            plt.interactive(True)
            plt.clf()
            plt.ylabel('Reward per episodes')
            plt.xlabel('Episodes')
            plt.plot(episode_rewards)
            plt.draw()

        print "[episode %d] episode reward: %f.  Epsilon now: %f" %\
            (i, episode_rewards[-1], agent.epsilon)

    return episode_rewards

def run_experiment(AgentClass, render):
    '''
    Runs the configured experiment for the given agent class.

    :param AgentClass: An RLAgent type to use to construct the agent.
    :param render: If true, use rendering.
    '''
    mdp = CliffMDP(12, 4, render=render)

    agent_args = {'gamma': mdp.gamma,
                  'alpha': 0.25,
                  'epsilon': 0.99}

    experiment_args = {'max_episodes': 500,
                       'epsilon_decay': 0.99,
                       'plot': True}

    agent = AgentClass(legal_actions=mdp.actions, **agent_args)
    rewards = train_agent(mdp, agent, **experiment_args)
    
    filename = "%s_rewards_%s_%s.pkl" % (AgentClass.__name__,\
        "_".join(["%s-%s" % (str(k), str(v)) for (k,v) 
            in agent_args.iteritems()]),
        "_".join(["%s-%s" % (str(k), str(v)) for (k,v) 
            in experiment_args.iteritems()]))
    
    f = open(filename, 'wb')
    pickle.dump(rewards, f)
    f.close()

    return rewards

def compare_sarsa_qlearning(render):
    '''
    The top-level method for running RL experiments.  Performs runs on the two
    defined agents, and plots the results.

    :param render: If true, render while training
    '''
    sarsa_rewards = run_experiment(SARSAAgent, render)
    qlearning_rewards = run_experiment(QLearningAgent, render)
    plt.interactive(False)
    plot_sarsa_vs_qlearning(sarsa_rewards, qlearning_rewards)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Executes RL experiments on the Cliff-World domain''')

    parser.add_argument('-r', '--render', action='store_true', 
        help="Toggle Tkinter rendering (off by default)", default=False)
    parser.add_argument('-k', '--keyboard', action='store_true', 
        help="Toggle keyboard mode (runs interactive episodes)", default=False)

    args = parser.parse_args()

    if args.keyboard:
        mdp = CliffMDP(12, 4, render=args.render)
        agent = SARSAAgent(legal_actions=mdp.actions, gamma=mdp.gamma)
        run_episode(mdp, agent, kbd_ctl=args.keyboard)
    else:
        compare_sarsa_qlearning(args.render)
