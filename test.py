# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:39:18 2019

@author: 42821
"""

import gym
                                 
env_name='AirRaid-ram-v0'
env = gym.make(env_name)
#env = gym.make('Acrobot-v1')

env.reset()
for i in range(1000):
    env.render()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print(i, action, next_state, reward, done, info)
env.close()