# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 15:22:16 2019

@author: you
"""

import numpy as np
import gym
import rlmodel

class Sample:
    def __init__(self,s,a,ns,r,critic=[0],next_critic=[0],td_error=0.5):
        self.state=s
        self.action=a
        self.next_state=ns
        self.reward=r
        self.critic=critic
        self.next_critic=next_critic
        self.td_error=td_error
        self.ma_td_err=0
    def __repr__(self):
        return str(["s:", '[%6.3f, %6.3f]' % tuple(self.state),  \
                    "a:", '[%6.3f]' % tuple(self.action), \
                    "n_s:", '[%6.3f, %6.3f]' % tuple(self.next_state), \
                    "r:", '[%6.3f]' % self.reward, \
                    "c:", '[%10.6f]' % tuple(self.critic), \
                    "n_c:",'[%10.6f]' % tuple(self.next_critic), \
                    "td_err:", '[%8.6f]' %  self.td_error , \
                    "ma_td_err:", '[%8.6f]' %  self.ma_td_err 
                    ]) + "\n"

    
class Experiment:
    def __init__(self, env_name='MountainCarContinuous-v0', model_name="basic", exp_params={}, net_params={}, env_params={}):
        
        self.replay_buffer = []
        self.replay_buffer_size = 40000
        self.best_buffer=[]
        self.best_buffer_size = 10000
        self.best_step = 1200
        
        self.best_ends = []
        self.best_ends_ratio  = 0.1
        self.segments_nb = 100
        self.segments_range=None
        self.full_filled = False
        
        self.minibatch_size = 64
        self.init_power_law_alpha = 0.6
        self.power_law_alpha = 0.6
        self.power_law_alpha_decenting_rate = 0.99
        self.min_power_law_alpha = 0.01
        
        self.distribution = None
        self.update_power_law_distribution(self.replay_buffer_size, self.segments_nb, self.minibatch_size, self.power_law_alpha)
        
        self.max_reward = -10000
        self.min_reward = 10000
        
        self.nb_experience = 10
        self.nb_episode = 300
        self.nb_step = 999
        
        self.exploration_rate = 0.9
        self.exploration_decent_rate = 0.98
        self.min_exploration_rate = 0.01
        self.exploration_episode = 100
        self.exploration_episode = 100
        self.success_exploration = 5
          
        self.render = False
        self.render_freq = 20
        
        self.show_states = False
        
        self.learn_best_rate = 0.1
        
        self.use_summary = False
        
        # Update Parameters
        for k, v in exp_params.items():
            setattr(self, k, v)
            print("update", k, "to", v)
                
        self.env = gym.make(env_name)
        self.model = rlmodel.RLModel(model_name, net_params, env_params)
        
        
    def update_power_law_distribution(self, batchsize, segments_nb, minibatch_size, alpha):
        distribution = [(1.0/i)**alpha for i in range(1,batchsize+1)]
        distribution = [sum(distribution[int(k*batchsize/segments_nb):int((k+1)*batchsize/segments_nb)]) for k in range(segments_nb)]
        s = sum(distribution)
        distribution = [i/s for i in distribution]
        self.distribution = distribution
            
    def random_select(self, replay_buffer, size):
        choices =  np.random.randint(0, len(replay_buffer), size)
        return [replay_buffer[i] for i in choices]
    
    
    def temporal_difference_error_sort(self, replay_buffer):
        return sorted(replay_buffer, key=lambda x:x.td_error*0.9+x.ma_td_err*0.1, reverse=True)
#        return sorted(replay_buffer, key=lambda x:x.td_error, reverse=True)
    
    def power_law_select(self, replay_buffer, minibatch_size):
        segments_range = self.segments_range
        segments_nb = self.segments_nb
        if segments_range==None:
            segments_range = [int(s*len(replay_buffer)/segments_nb) for s in range(segments_nb)]
        elif len(replay_buffer) < minibatch_size and len(replay_buffer)%10000 <1500:
            segments_range = [int(s*len(replay_buffer)/segments_nb) for s in range(segments_nb)]
        elif len(replay_buffer)==minibatch_size and not self.full_filled:
            segments_range = [int(s*minibatch_size/segments_nb) for s in range(segments_nb)]
            self.full_filled = True
            
            
        selections = []
        for i in range(minibatch_size): 
            segment_index= min(np.random.choice(range( segments_nb ),1, p=self.distribution)[0], segments_nb-2)
        #    segment_index=0   
            inf2 = segments_range[segment_index]
            sup2 = min(segments_range[segment_index+1], len(replay_buffer)-1)
            selections.append(replay_buffer[np.random.randint(int(inf2), int(sup2))])
    #    print (selections)
        return selections
     
    def best_select(self, replay_buffer, best_buffer, minibatch_size, best_selection_ratio=0.0, best_ends=[], best_ends_ratio=0.0):
        best_ends_size = int(minibatch_size * best_ends_ratio)
        best_selection_size = int(minibatch_size * best_selection_ratio)
        p_l_size = minibatch_size - best_ends_size - best_selection_size
        
        samples = self.power_law_select(replay_buffer, p_l_size)
        
        if len(best_buffer) > 0 and best_selection_ratio > 0:  
            bb = self.random_select(best_buffer, best_selection_size) 
            samples.extend(bb) 
        if len(best_ends) > 0 and best_ends_ratio > 0:
            be = self.random_select(best_ends, best_ends_size) 
            samples.extend(be)
        return samples
        
    
    def random_minibatch(self, samples, size):
        if np.random.uniform()<0.3: 
    #        print(bests)
            return np.random.choice(self.bests, size)
        np.random.shuffle(samples)
        return samples[:size]
     
    def get_ecalated_normal_reward(self, reward): 
        if self.max_reward - self.min_reward == 0:
            return -1
        return (reward - self.min_reward) / (self.max_reward - self.min_reward) * 2 - 1
        
    def get_polar_normal_reward(self, reward): 
        if reward<0:
            return reward/self.min_reward*(-1) 
        elif reward>0:
            return reward/self.max_reward*1
        else :
            return 0  
        
    def get_normal_reward(self, reward): 
        if self.max_reward - self.min_reward == 0:
            return 0
        return reward / (self.max_reward - self.min_reward)
    
    def start(self):
        for i_exp in range(self.nb_experience):
            print("###DDPG Training", i_exp, " Experience starts")
                  
            self.model.init_session()
            
            total_step = 0
            replay_buffer = []
            

            self.max_reward = -10000
            self.min_reward = 10000
            self.segments_range = None
            self.full_filled = False 
            
        
            for i_episode in range(self.nb_episode):
                
                # Get states from Eviroments
                state = self.env.reset() 
                state = np.reshape(state, newshape=([2]))
                temp_batch = []
                reward_sum = 0 # Initial rewrard, to show at the end.
                loss_sum = 0
                
                if i_episode > 20:
                    #Decending exploration_rate, after 10 times, smaller than 0.005
                    self.exploration_rate = max(self.exploration_rate*self.exploration_decent_rate, \
                                            self.min_exploration_rate)
                
                for t in range(self.nb_step):
                    
                    #Choose action
                    if np.random.random() < self.exploration_rate:
                        action = self.env.action_space.sample()*100
                    else:
                        action = self.model.get_action(state)
                        
                    action = np.reshape(action, newshape=([1]))
                    action = self.get_noised_action(action)
                    
                    
                    #Get next state, rewoard, if it is done
                    next_state, reward, done, info = self.env.step(action)
                    next_state = np.reshape(next_state, newshape=([2]))
                    
                    #Cumulate the reward
                    reward_sum += reward
                    #Create samples object and save it in the sample batch
                    sample = Sample(s=state,a=action,ns=next_state,r=reward)
                    temp_batch.append(sample)
                    state=next_state
                     
                    # Update the reward ever seen
                    if self.max_reward < reward:
                        self.max_reward = reward
                    if self.min_reward > reward:
                        self.min_reward = reward
                        
                    
                    if i_episode > 20:
                        #Selecting samples
                        training_samples =  self.power_law_select(replay_buffer, self.minibatch_size)
                        
                        #Train the model with samples
                        loss_value = self.train_model(training_samples)
                        
                        loss_sum += loss_value
                        
#                        if t % 300 == 0:
#                            print(training_samples)
                        
                        # Sort the sample batch
                        if t % 100 == 0:
                            replay_buffer = self.temporal_difference_error_sort(replay_buffer)
                            
                    if done and t+1<999:
                        break
                
                    
                if i_episode % self.render_freq == 0 and self.render:
                    self.env.render()
                    
                total_step += (t+1)
                print('Episode ', i_episode, 'Nb Steps ', t+1, "\t",
                      "Rewards : ", int(reward_sum), "\t",
                      "Avg Loss:", np.round(loss_sum / t, 3), "\t", #t is Step
                      "Exploration r:", np.round(self.exploration_rate, 3), "\t",
                      "Total Steps :", total_step)
                
                # Store the sample       
                self.store_sample(temp_batch, replay_buffer)
                self.model.decent_training_rate(decent_rate=0.95)
  
    def start_best_reward(self):
        for i_exp in range(self.nb_experience):
            print("###DDPG Training", i_exp, " Experience starts")
                  
            self.model.init_session()
            
            total_step = 0
            self.replay_buffer = []
            
            self.best_buffer = []
            
            best_reward = -999
            
            current_exploration_rate = self.exploration_rate
            current_power_law_alpha = self.power_law_alpha

            self.max_reward = -10000
            self.min_reward = 10000
            self.segments_range = None
            self.full_filled = False
            
            success_times = 0
        
        
#            self.warm_up(self.exploration_episode)
        
        
            for i_episode in range(self.nb_episode):
                
                # Get states from Eviroments
                state = self.env.reset() 
                state = np.reshape(state, newshape=([2]))
                temp_batch = []
                reward_sum = 0 # Initial rewrard, to show at the end.
                loss_sum = 0
                 
                #Decending exploration_rate, after 10 times, smaller than 0.005
                current_exploration_rate = max(current_exploration_rate*self.exploration_decent_rate, \
                                        self.min_exploration_rate)
                
                for t in range(self.nb_step):
                    
                    #Choose action
                    if np.random.random() < current_exploration_rate:
                        action = self.get_sample_action(self.model.action_size, 2)
                    else:
                        action = self.model.get_action(state)
                        
                    action = np.reshape(action, newshape=([1]))
                    action = self.get_noised_action(action)
                    
                    
                    #Get next state, rewoard, if it is done
                    next_state, reward, done, info = self.env.step(action)
                    next_state = np.reshape(next_state, newshape=([2]))
                    
                    #Cumulate the reward
                    reward_sum += reward
                    #Create samples object and save it in the sample batch
                    sample = Sample(s=state, a=action, ns=next_state, r=reward)
                    temp_batch.append(sample)
                    state=next_state
                     
                    # Update the reward ever seen
                    if self.max_reward < reward:
                        self.max_reward = reward
                    if self.min_reward > reward:
                        self.min_reward = reward
                    
                     
                    #Selecting samples
                    training_samples =  self.best_select(self.replay_buffer, self.best_buffer, self.minibatch_size, self.learn_best_rate)
                    
                    #Train the model with samples
                    loss_value = self.train_model(training_samples) 
                    loss_sum += loss_value
                        
                    if t == 0:
                        print(sorted(training_samples, key=lambda x:x.critic, reverse=True)[:20])
#                        
                    # Sort the sample batch
                    if t % 100 == 0:
                        self.replay_buffer = self.temporal_difference_error_sort(self.replay_buffer) 
                            
                    if done and t+1<999:
                        success_times += 1
                        break
                     
                if i_episode % self.render_freq == 0 and self.render:
                    self.env.render()
                    
                total_step += (t+1)
                print('Episode ', i_episode, 'Nb Steps ', t+1, "\t",
                      "Rewards : ", int(reward_sum), "\t",
                      "Avg Loss:", np.round(loss_sum / t, 3), "\t", #t is Step
                      "Exploration r:", np.round(current_exploration_rate, 3), "\t",
                      "C Learning r", np.round(self.model.get_critic_learning_rate(), 7),"\t",
                      "Power Law a", np.round(current_power_law_alpha, 3),"\t",
                      "Sucess",success_times,"\t",
                      "Total Steps :", total_step)
                
                # Store the sample       
                self.store_sample(temp_batch, self.replay_buffer, self.replay_buffer_size)
                if done: 
                    new_best_rewards = [x for x in temp_batch if x.reward > best_reward * 0.99]
                    if new_best_rewards:
                        max_best = max(new_best_rewards, key=lambda x:x.reward)
                        best_reward = max(best_reward, max_best.reward)
                        
                    self.store_sample(new_best_rewards, self.best_buffer, self.best_buffer_size)
                    print(new_best_rewards)
                # decent learning Rate 
                self.model.decent_training_rate()
                current_power_law_alpha = max(self.min_power_law_alpha, 
                                           current_power_law_alpha * self.power_law_alpha_decenting_rate)
                self.update_power_law_distribution(self.replay_buffer_size, self.segments_nb, self.minibatch_size, current_power_law_alpha)
                    
    def start_with_best_batch(self):
        for i_exp in range(self.nb_experience):
            print("###DDPG Training", i_exp, " Experience starts")
                  
            self.model.init_session()
            
            total_step = 0
            self.replay_buffer = []
            
            self.best_buffer = []
            self.best_step = 1200
            
            current_exploration_rate = self.exploration_rate
            
            current_power_law_alpha = self.power_law_alpha

            self.max_reward = -10000
            self.min_reward = 10000
            self.segments_range = None
            self.full_filled = False
            
            success_times = 0
            
            self.warm_up(self.exploration_episode)
        
            for i_episode in range(self.nb_episode):
                
                # Get states from Eviroments
                state = self.env.reset() 
                state = np.reshape(state, newshape=([2]))
                temp_batch = []
                reward_sum = 0 # Initial rewrard, to show at the end.
                loss_sum = 0
              
                #Decending exploration_rate, after 10 times, smaller than 0.005
                current_exploration_rate = max(current_exploration_rate*self.exploration_decent_rate, \
                                        self.min_exploration_rate)
                
                for t in range(self.nb_step):
                    
                    #Choose action
                    if np.random.random() < current_exploration_rate:
                        action = self.get_sample_action(self.model.action_size, 2)
                    else:
                        action = self.model.get_action(state)
                        
                    action = np.reshape(action, newshape=([1]))
                    action = self.get_noised_action(action)
                    
                    
                    #Get next state, rewoard, if it is done
                    next_state, reward, done, info = self.env.step(action)
                    next_state = np.reshape(next_state, newshape=([2]))
                    
                    #Cumulate the reward
                    reward_sum += reward
                    #Create samples object and save it in the sample batch
                    sample = Sample(s=state, a=action, ns=next_state, r=reward)
                    temp_batch.append(sample)
                    state=next_state
                     
                    # Update the reward ever seen
                    if self.max_reward < reward:
                        self.max_reward = reward
                    if self.min_reward > reward:
                        self.min_reward = reward
                     
                    #Selecting samples
                    training_samples =  self.best_select(self.replay_buffer, self.best_buffer, self.minibatch_size, self.learn_best_rate)
                    
                    #Train the model with samples
                    loss_value = self.train_model(training_samples) 
                    loss_sum += loss_value
                    
                    if t == 0 and self.show_states:
                        print(sorted(training_samples, key=lambda x:x.td_error*0.8+x.ma_td_err*0.2, reverse=True)[:20])

                    # Sort the sample batch
                    if t % 20 == 0 and np.random.random() < self.power_law_alpha:
                        self.replay_buffer = self.temporal_difference_error_sort(self.replay_buffer) 
                            
                    if done and t+1<999: 
                        success_times += 1
                        break
                     
                    if i_episode % self.render_freq == 1 and self.render:
                        self.env.render()
                    
                total_step += (t+1)
                print('Episode: [%4d]' % i_episode, 
                      'Nb Steps: [%4d]' % (t + 1),
                      "Rewards: [%4d]" % int(reward_sum),
                      "Avg Loss: [%9.3f]" % (loss_sum / t), #t is Step
                      "Exploration r: [%5.3f]" % current_exploration_rate,
                      "C Learning r:  [%9.7f]" % self.model.get_critic_learning_rate(),
                      "Power Law a: [%5.3f]" % current_power_law_alpha,
                      "Sucess: [%3d]" % success_times,
                      "Total Steps: [%8d]" % total_step)
                
                # Store the sample       
                self.store_sample(temp_batch, self.replay_buffer, self.replay_buffer_size)
                if done and t+1 < self.nb_step and t+1 < self.best_step + 100 or i_episode == 0:
                    self.best_step = t+1
                    self.store_sample(temp_batch, self.best_buffer, self.best_buffer_size)
                    
                if self.use_summary:
                    writing_samples = self.best_select(replay_buffer=self.replay_buffer, 
                                                         best_buffer=self.best_buffer, 
                                                         minibatch_size=1000, 
                                                         best_selection_ratio=self.learn_best_rate, 
                                                         best_ends=self.best_ends, 
                                                         best_ends_ratio=self.best_ends_ratio)
                    self.write_summaries(writing_samples, i_episode, int(reward_sum),  (t + 1))
                    
                # decent learning Rate 
                self.model.decent_training_rate()
                
                # decent power law alpha and update the distribution
                current_power_law_alpha = max(self.min_power_law_alpha, 
                                           current_power_law_alpha * self.power_law_alpha_decenting_rate) 
                self.update_power_law_distribution(self.replay_buffer_size, self.segments_nb, self.minibatch_size, current_power_law_alpha)
                
                
    def start_with_bests_2(self):
        for i_exp in range(self.nb_experience):
            print("###DDPG Training", i_exp, " Experience starts")
                  
            self.model.init_session()
            
            total_step = 0
            self.replay_buffer = []
            
            self.best_buffer = []
            self.best_step = 1200
            
            self.best_ends = []
            
            current_exploration_rate = self.exploration_rate
            
            current_power_law_alpha = self.power_law_alpha

            self.max_reward = -10000
            self.min_reward = 10000
            self.segments_range = None
            self.full_filled = False
            
            success_times = 0
            
            self.warm_up(self.exploration_episode)
        
            for i_episode in range(self.nb_episode):
                
                # Get states from Eviroments
                state = self.env.reset() 
                state = np.reshape(state, newshape=([2]))
                temp_batch = []
                reward_sum = 0 # Initial rewrard, to show at the end.
                loss_sum = 0
              
                #Decending exploration_rate, after 10 times, smaller than 0.005
                current_exploration_rate = max(current_exploration_rate*self.exploration_decent_rate, \
                                        self.min_exploration_rate)
                
                for t in range(self.nb_step):
                    
                    #Choose action
                    if np.random.random() < current_exploration_rate:
                        action = self.get_sample_action(self.model.action_size, 2)
                    else:
                        action = self.model.get_action(state)
                        
                    action = np.reshape(action, newshape=([1]))
                    action = self.get_noised_action(action)
                    
                    
                    #Get next state, rewoard, if it is done
                    next_state, reward, done, info = self.env.step(action)
                    next_state = np.reshape(next_state, newshape=([2]))
                    
                    #Cumulate the reward
                    reward_sum += reward
                    #Create samples object and save it in the sample batch
                    sample = Sample(s=state, a=action, ns=next_state, r=reward)
                    temp_batch.append(sample)
                    state=next_state
                     
                    # Update the reward ever seen
                    if self.max_reward < reward:
                        self.max_reward = reward
                    if self.min_reward > reward:
                        self.min_reward = reward
                        
                    #Selecting samples 
                    training_samples =  self.best_select(replay_buffer=self.replay_buffer, 
                                                         best_buffer=self.best_buffer, 
                                                         minibatch_size=self.minibatch_size, 
                                                         best_selection_ratio=self.learn_best_rate, 
                                                         best_ends=self.best_ends, 
                                                         best_ends_ratio=self.best_ends_ratio)
                    
                    #Train the model with samples
                    loss_value = self.train_model(training_samples)
                    loss_sum += loss_value
                    
                    if t == 0 and self.show_states:
                        print(sorted(training_samples, key=lambda x:x.td_error*0.9+x.ma_td_err*0.1, reverse=True)[:20])
#                        print(sorted(training_samples, key=lambda x:x.td_error, reverse=True)[:20])

                    # Sort the sample batch
                    if t % 20 == 0 and np.random.random() < self.power_law_alpha:
                        self.replay_buffer = self.temporal_difference_error_sort(self.replay_buffer) 
                            
                    if done and t+1<999: 
                        success_times += 1
                        self.best_ends.append(sample)
                        break
                     
                    if i_episode % self.render_freq == 1 and self.render:
                        self.env.render()
                    
                total_step += (t+1)
                print('Episode: [%4d]' % i_episode, 
                      'Nb Steps: [%4d]' % (t + 1),
                      "Rewards: [%4d]" % int(reward_sum),
                      "Avg Loss: [%9.3f]" % (loss_sum / t), #t is Step
                      "Exploration r: [%5.3f]" % current_exploration_rate,
                      "C Learning r:  [%9.7f]" % self.model.get_critic_learning_rate(),
                      "Power Law a: [%5.3f]" % current_power_law_alpha,
                      "Sucess: [%3d]" % success_times,
                      "Total Steps: [%8d]" % total_step)
                
                # Store the sample       
                self.store_sample(temp_batch, self.replay_buffer, self.replay_buffer_size)
                if done and t+1 < self.nb_step and t+1 < self.best_step + 100 or i_episode == 0:
                    self.best_step = t+1
                    self.store_sample(temp_batch, self.best_buffer, self.best_buffer_size)
                    
                if self.use_summary:
                    writing_samples =  self.best_select(self.replay_buffer, self.best_buffer, 1000, self.learn_best_rate)
                    self.write_summaries(writing_samples, i_episode, int(reward_sum),  (t + 1))
                    
                # decent learning Rate 
                self.model.decent_training_rate()
                
                # decent power law alpha and update the distribution
                current_power_law_alpha = max(self.min_power_law_alpha, 
                                           current_power_law_alpha * self.power_law_alpha_decenting_rate) 
                self.update_power_law_distribution(self.replay_buffer_size, self.segments_nb, self.minibatch_size, current_power_law_alpha)
                
                
                
    def start_with_warmup(self):
        for i_exp in range(self.nb_experience):
            print("###DDPG Training", i_exp, " Experience starts")
                  
            self.model.init_session()
            
            total_step = 0
            self.replay_buffer = []
            
            self.exploration_episode = 100
            current_exploration_rate = self.exploration_rate

            self.max_reward = -10000
            self.min_reward = 10000
            self.segments_range = None
            self.full_filled = False
            
            success_times = 0
            
            #Warm Up
            print("Warm up")
            for i_episode in range(self.exploration_episode):
                state = self.env.reset() 
                state = np.reshape(state, newshape=([2]))
                temp_batch = []
                reward_sum = 0 # Initial rewrard, to show at the end.
                loss_sum = 0
                
                for t in range(self.nb_step):
                    
                    #Choose action
                    action = self.get_sample_action(self.model.action_size, 2)
                        
                    action = np.reshape(action, newshape=([self.model.action_size]))
                    action = self.get_noised_action(action)
                    
                    
                    #Get next state, rewoard, if it is done
                    next_state, reward, done, info = self.env.step(action)
                    next_state = np.reshape(next_state, newshape=([2]))
                    
                    #Cumulate the reward
                    reward_sum += reward
                    #Create samples object and save it in the sample batch
                    sample = Sample(s=state, a=action, ns=next_state, r=reward)
                    temp_batch.append(sample)
                    state=next_state
                     
                    # Update the reward ever seen
                    if self.max_reward < reward:
                        self.max_reward = reward
                    if self.min_reward > reward:
                        self.min_reward = reward
                     
                            
                    if done and t+1<999:
                        break 
                
                # Store the sample       
                self.store_sample(temp_batch, self.replay_buffer, self.replay_buffer_size) 
                
                
        
            print("Warm up Finished", "Replay Buffer Size:",len(self.replay_buffer))
            for i_episode in range(self.nb_episode):
                
                # Get states from Eviroments
                state = self.env.reset() 
                state = np.reshape(state, newshape=([2]))
                temp_batch = []
                reward_sum = 0 # Initial rewrard, to show at the end.
                loss_sum = 0
                 
                #Decending exploration_rate, after 10 times, smaller than 0.005
                current_exploration_rate = max(current_exploration_rate*self.exploration_decent_rate, \
                                            self.min_exploration_rate)
                
                for t in range(self.nb_step):
                    
                    #Choose action
                    if np.random.random() < current_exploration_rate:
                        action = self.get_sample_action(self.model.action_size, 2)
                    else:
                        action = self.model.get_action(state)
                        
                    action = np.reshape(action, newshape=([1]))
                    action = self.get_noised_action(action)
                    
                    
                    #Get next state, rewoard, if it is done
                    next_state, reward, done, info = self.env.step(action)
                    next_state = np.reshape(next_state, newshape=([2]))
                    
                    #Cumulate the reward
                    reward_sum += reward
                    
                    #Create samples object and save it in the sample batch
                    sample = Sample(s=state, a=action, ns=next_state, r=reward)
                    temp_batch.append(sample)
                    state=next_state
                     
                    # Update the reward ever seen
                    if self.max_reward < reward:
                        self.max_reward = reward
                    if self.min_reward > reward:
                        self.min_reward = reward
                     
                    #Selecting samples
                    training_samples =  self.power_law_select(self.replay_buffer, self.minibatch_size)
                    
                    #Train the model with samples
                    loss_value = self.train_model(training_samples) 
                    
                    loss_sum += loss_value
                    
#                    if t == 0:
#                        print(sorted(training_samples, key=lambda x:x.td_error*0.8+x.ma_td_err*0.2, reverse=True)[:20])
#                        
                    # Sort the sample batch
                    if t % 20 == 1:
                        self.replay_buffer = self.temporal_difference_error_sort(self.replay_buffer) 
                            
                    if done and t+1<999:
                        success_times += 1
                        break
                     
                if i_episode % self.render_freq == 1 and self.render:
                    self.env.render()
                    
                total_step += (t+1)
                print('Episode ', i_episode, 'Nb Steps ', t+1, "\t",
                      "Rewards : ", int(reward_sum), "\t",
                      "Avg Loss:", np.round(loss_sum / t, 3), "\t", #t is Step
                      "Exploration r:", np.round(current_exploration_rate, 3), "\t",
                      "Critic Learning r", np.round(self.model.get_critic_learning_rate(), 7),"\t",
                      "Sucess",success_times,"\t",
                      "Total Steps :", total_step)
                
                # Store the sample       
                self.store_sample(temp_batch, self.replay_buffer, self.replay_buffer_size)
                    
                # decent learning Rate 
                self.model.decent_training_rate()
                self.power_law_alpha = max(self.min_power_law_alpha, 
                                           self.power_law_alpha * self.power_law_alpha_decenting_rate) 
    
    def warm_up(self, warm_up_episode):
        #Warm Up
        print("Warm up")
        
        success_rest = self.success_exploration 
#        for i_episode in range(warm_up_episode):
        while success_rest > 0:
            state = self.env.reset() 
            state = np.reshape(state, newshape=([2]))
            temp_batch = [] 
            
            for t in range(self.nb_step):
                
                #Choose action
                action = self.get_sample_action(self.model.action_size, 2)
                    
                action = np.reshape(action, newshape=([self.model.action_size]))
                action = self.get_noised_action(action)
                
                
                #Get next state, rewoard, if it is done
                next_state, reward, done, info = self.env.step(action)
                next_state = np.reshape(next_state, newshape=([2]))
                 
                #Create samples object and save it in the sample batch
                sample = Sample(s=state, a=action, ns=next_state, r=reward)
                temp_batch.append(sample)
                state=next_state
                 
                # Update the reward ever seen
                if self.max_reward < reward:
                    self.max_reward = reward
                if self.min_reward > reward:
                    self.min_reward = reward
                 
                if reward > 0:
                    self.best_ends.append(sample) 
                        
                if done and t+1 < 999:
                    success_rest -= 1
                    self.store_sample(temp_batch, self.best_buffer, self.best_buffer_size)
                    break 
        
            # Store the sample       
            self.store_sample(temp_batch, self.replay_buffer, self.replay_buffer_size) 
#            if done and t+1 < self.nb_step and t+1 < best_step + 100 or i_episode == 0:
#            if done and t+1 < self.nb_step and t+1 < best_step + 100:
#                best_step = t+1
#                self.store_sample(temp_batch, self.best_buffer, self.best_buffer_size)
        print("Warm up Finished", 
              "Replay Buffer Size:",len(self.replay_buffer),
              "Best Buffer Size:",len(self.best_buffer), 
              "Best ends Size:",len(self.best_ends), 
              )
                                
    def train_model(self, samples):
        next_state_samples=[]
        state_samples=[]
        action_samples=[]
        reward_samples=[]
        for s in samples:
            next_state_samples.append(s.next_state)
            action_samples.append(s.action)
            state_samples.append(s.state)
            reward_samples.append([self.get_polar_normal_reward(s.reward)])
#            reward_samples.append([s.reward])
            
        loss_value = self.model.training(state_samples, action_samples, next_state_samples,reward_samples,samples)
        return loss_value 
    
    def write_summaries(self, samples, i_episode, current_reward, current_steps):
        next_state_samples=[]
        state_samples=[]
        action_samples=[]
        reward_samples=[]
        for s in samples:
            next_state_samples.append(s.next_state)
            action_samples.append(s.action)
            state_samples.append(s.state)
            reward_samples.append([s.reward])
        self.model.write_summaries(state_samples, action_samples,next_state_samples,reward_samples, i_episode, current_reward, current_steps)
    
    def get_noised_action(self, action, noise=0.2):
        action = min(max(-1, action[0]), 1)
        return [min(max(-1, action+np.random.uniform(-noise, noise)), 1)]
    
    def get_sample_action(self, size, border=2):
        action = list(np.random.random(size) * border)
        return action
    
    
    def store_sample(self, temp_batch, replay_buffer, replay_buffer_size):
        if replay_buffer_size > len(temp_batch) + len(replay_buffer):
            replay_buffer.extend(temp_batch)
        elif replay_buffer_size == len(replay_buffer):
            for s in temp_batch:
                replay_buffer[np.random.randint(len(replay_buffer)//10, len(replay_buffer))] = s
        else:
            while replay_buffer_size > len(replay_buffer):
                replay_buffer.append(temp_batch.pop())
            for s in temp_batch:
                replay_buffer[np.random.randint(len(replay_buffer)//10, len(replay_buffer))] = s
                
    #no need to do this
    def update_td_error(self, training_samples):
        for s in training_samples:
            s.critic, s.next_critic, s.td_error = \
                self.model.get_td_error(s.action, s.state, s.next_state, self.get_normal_reward(s.reward))



        