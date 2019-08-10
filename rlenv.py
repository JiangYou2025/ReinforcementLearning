# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:05:00 2019

@author: 42821
"""
import gym

class Enviroment:
    def __init__(self, env_name):
        self.continous_action = True
        self.continous_state = True
        self.env_name = env_name
        self.env = gym.makeEnv(env_name)
        self.use_image_state = False
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def sample_action(self):
        return self.env.action_space.sample()
    
    def reset(self):
        self.env.reset()

    def random_action_discrete(self):
        pass
    def random_action_continous(self, max_range):
        pass
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()