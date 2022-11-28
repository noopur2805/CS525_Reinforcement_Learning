#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import random
import numpy as np

"""
    Monte-Carlo
    In this problem, you will implement an AI player for Blackjack.
    The main goal of this problem is to get familiar with Monte-Carlo algorithm.
    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.

    You don't have to follow the comments to write your code. They are provided
    as hints in case you need.
"""


def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and hits otherwise

    Parameters:
    -----------
    observation

    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """

    score, dealer_score, usable_ace = observation
    if score >= 20:
        return 0
    else:
        return 1


def mc_prediction(policy, env, n_episodes, gamma=1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an observation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value

    Note: at the beginning of each episode, you need to initialize the environment using env.reset()
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    ############################
	# YOUR IMPLEMENTATION HERE #

	# loop each episode
    for k in range(n_episodes):
        # initialize the episode
        current_state = env.reset()  

        # initialize empty episode list
        episode = []
        done = False

        # loop until episode generation is done
        while not done:

            # select an action
            action = policy(current_state)  

            # return a reward and new state
            new_state, reward, done, info, prob = env.step(action)  

            # append state, action, reward to episode
            episode.append((current_state, action, reward)) 

            # update state to new state
            current_state = new_state  
        
        state_returns = []  
        G = 0

        # loop for each step of episode, t = T-1, T-2,...,0
        for (state, action, reward) in reversed(episode):
            # compute G
            G = gamma*G + reward
            state_returns.append(G)

        state_returns.reverse()  

        states_visited = []
        for index, (observation, action, reward) in enumerate(episode):
            # unless state_t appears in states
            if observation not in states_visited:

                # update return_count
                returns_count[observation] += 1

                # update return_sum
                returns_sum[observation] += state_returns[index]

                # calculate average return for this state over all sampled episodes
                V[observation] = returns_sum[observation]/returns_count[observation]
                
                states_visited.append(observation)

    return V


def epsilon_greedy(Q, state, nA, epsilon=0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    state_action_probability = Q[state]
    greedy_policy_index = np.argmax(state_action_probability)
    prob = np.ones(nA, float) * (epsilon / nA)  
    prob[greedy_policy_index] = (epsilon / nA) + 1 - epsilon  

    action = np.random.choice(np.arange(len(state_action_probability)), p=prob)
    return action

def mc_control_epsilon_greedy(env, n_episodes, gamma=1.0, epsilon=0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.array(nA)
    action_count = env.action_space.n
    Q = defaultdict(lambda: np.zeros(action_count))

    ############################
	# YOUR IMPLEMENTATION HERE #
    
    for i in range(n_episodes):

        # define decaying epsilon
        if epsilon <= 0.05:
            epsilon = 0.05 # Threshold epsilon to prevent it becoming negative
        else:
            epsilon = epsilon - (0.1 / n_episodes)

        # initialize the episode
        current_state = env.reset()  

        # generate empty episode list
        episode = []
        done = False

        # loop until one episode generation is done
        while not done:

            # get an action from epsilon greedy policy
            action = epsilon_greedy(Q, current_state, action_count, epsilon) 

            # return a reward and new state
            new_state, reward, done, info, prob = env.step(action)  

            # append state, action, reward to episode
            episode.append((current_state, action, reward))  # append state, action, reward to episode

            # update state to new state
            current_state = new_state  

        state_action_returns = []  # G for each state
        G = 0

        # loop for each step of episode, t = T-1, T-2, ...,0
        for (state, action, reward) in reversed(episode):

            # compute G
            G = gamma * G + reward
            state_action_returns.append(G)

        state_action_returns.reverse()  

        visited = []

        # unless the pair state_t, action_t appears in <state action> pair list
        for index, (state, action, reward) in enumerate(episode):
            if (state, action) not in visited:
                
                # update return_count
                returns_count[(state, action)] += 1

                # update return_sum
                returns_sum[(state, action)] += state_action_returns[index]

                # calculate average return for this state over all sampled episodes
                Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]

                visited.append((state, action))

    return Q