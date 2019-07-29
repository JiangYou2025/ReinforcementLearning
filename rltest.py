# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 19:15:29 2019

@author: you
"""

import rloutils
net_params={
    "gamma": 0.99,
    'critic_learning_rate': 0.000003,
    'min_c_learning_rate':  0.000001,
    "critic_decent_rate": 0.985,
    'actor_learning_rate': 0.0000001,
    "hidden_layer_size": 80,
    "use_summary": False
}
exp_params={
    'exploration_rate': 0.9,
    'exploration_decent_rate': 0.92,
    'learn_best_rate':0.2,
    'best_ends_ratio':0.05,
    'min_exploration_rate': 0.01,
    "render": False,
    "render_freq": 10,
    "show_states": True,
    "nb_episode":1000,
    "replay_buffer_size":40000,
    "exploration_episode":100,
    "power_law_alpha_decenting_rate":0.99,
    "use_summary": True,
    "success_exploration":5,
    
}
experiment = rloutils.Experiment(net_params=net_params, 
                                 exp_params=exp_params)

experiment.start_with_bests_2()
