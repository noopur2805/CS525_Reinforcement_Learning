#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from collections import namedtuple
import math

import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd 

from agent import Agent
from dqn_model import DQN
import time

import os

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)
        
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

GAMMA = 0.99
EPSILON = 0.99
EPS_START = EPSILON
EPS_END = 0.005
EPS_DECAY = 1000
batch_size = 32
LR = 1e-5
TARGET_UPDATE = 20000

# Parameters for Replay Buffer
CAPACITY = 500 
memory = deque(maxlen=CAPACITY)
allEpsilons = []
learningThreshold = 100
LOAD = True


class Agent_DQN():
    def __init__(self, env, args):
        # Parameters for q-learning

        super(Agent_DQN,self).__init__()

        self.env = env
        state = env.reset()
        state = state.transpose(2,0,1)

        self.policy_net = DQN(state.shape, self.env.action_space.n) 
        self.target_net = DQN(state.shape, self.env.action_space.n) 
        self.target_net.load_state_dict(self.policy_net.state_dict()) 

        if USE_CUDA:
            print("Using CUDA . . .     ")
            self.policy_net = self.policy_net.cuda()
            self.target_net = self.target_net.cuda()

        print('Nnetwork initialized')

        if args.test_dqn or LOAD == True:
            print('loading model')
            checkpoint = torch.load('model.pth')
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
            
    def init_game_setting(self):
        print('loading model')
        checkpoint = torch.load('model.pth')
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])    
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        memory.append((state, action, reward, next_state, done))
    
    def replay_buffer(self):
        state, action, reward, next_state, done = zip(*random.sample(memory, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)
    
    def make_action(self, observation, test=True):

        observation = observation.transpose(2,0,1)
        if np.random.random() > EPSILON or test==True:
            observation   = Variable(torch.FloatTensor(np.float32(observation)).unsqueeze(0), volatile=True)
            q_value = self.policy_net.forward(observation)
            action  = q_value.max(1)[1].data[0]
            action = int(action.item())            
        else:
            action = random.randrange(4)
        return action

    def optimize_model(self):

        states, actions, next_states, rewards, dones  = self.replay_buffer()

        states_tensor = Variable(torch.FloatTensor(np.float32(states)))
        next_states_tensor = Variable(torch.FloatTensor(np.float32(next_states)), volatile=True)
        actions_tensor = Variable(torch.LongTensor(actions))
        rewards_tensor = Variable(torch.FloatTensor(rewards))
        done = Variable(torch.FloatTensor(dones))

        state_action_values = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        next_state_values = self.target_net(next_states_tensor).max(1)[0]
        expected_q_value = rewards_tensor + next_state_values * GAMMA * (1 - done) #+ rewards_tensor
        loss = (state_action_values - Variable(expected_q_value.data)).pow(2).mean()
        return loss
        
        
    def train(self):
        optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        AvgScore = 0
        AvgRewards = []
        AllScores = []
        episode = 0

        while ((AvgScore < 50) and (episode <= 100000)) :
                     
            state = self.env.reset()
            done = False
            EpisodeReward = 0
            tBegin = time.time()
            done = False

            while not done:
                action = self.make_action(state)    
                nextState, reward, done, _ = self.env.step(action)
                self.push(state.transpose(2,0,1), action, nextState.transpose(2,0,1), reward, done)

                state = nextState   
                if len(memory) > learningThreshold:
                    loss = self.optimize_model()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    episode = 0
                    continue        

                # Update exploration factor
                EPSILON = EPS_START + (EPS_START - EPS_END) * math.exp(-(episode+1) * EPS_DECAY)
                allEpsilons.append(EPSILON)
                EpisodeReward += reward

                if (episode+1) % TARGET_UPDATE == 0:
                    print('Updating Network')
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            episode += 1
            AllScores.append(EpisodeReward)
            AvgScore = np.mean(AllScores[-100:])
            AvgRewards.append(AvgScore)
            
            if len(memory) > learningThreshold: 
                print('Episode: ', episode, ' score:', EpisodeReward, ' Avg Score:',AvgScore,' epsilon: ', EPSILON, ' t: ', time.time()-tBegin, ' loss:', loss.item())

            if episode % 500 == 0:
                torch.save({
                    'epoch': episode,
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'AvgRewards': AvgRewards
                }, 'model.pth')

                with open('rewards.csv', mode='w') as dataFile:
                    rewardwriter = csv.writer(dataFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    rewardwriter.writerow(AvgRewards)

        print('Completed')
        torch.save({
            'epoch': episode,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'AvgRewards': AvgRewards
        }, 'model.pth')

        with open('rewards.csv', mode='w') as dataFile:
            rewardwriter = csv.writer(dataFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            rewardwriter.writerow(AvgRewards)
