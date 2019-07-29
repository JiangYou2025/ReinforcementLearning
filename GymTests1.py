# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:04:33 2018

@author: you
"""
import numpy as np
import gym
env = gym.make('MountainCarContinuous-v0')

class Sample:
    def __init__(self,s,a,ns,r,critic=None,next_critic=None,td_error=0.001):
        self.state=s
        self.action=a
        self.next_state=ns
        self.reward=r
        self.critic=critic
        self.next_critic=next_critic
        self.td_error=td_error
    def __repr__(self):
        return str({"state":str(np.round(self.state,3)),'action':str(np.round(self.action,3)),
                    "next_state": str(np.round(self.next_state,3)), "reward":str(np.round(self.reward,3)),
                    "critic":str(np.round(self.critic,3)),"next_critic":str(np.round(self.next_critic,3)),
                    "td_error":str(np.round(self.td_error,3))})

sample_batch = []
batchsize=40000
segments_nb=100

minibatch_size=64
alpha=0.6

distribution=[(1.0/i)**alpha for i in range(1,batchsize+1)]
distribution=[sum(distribution[int(k*batchsize/segments_nb):int((k+1)*batchsize/segments_nb)]) for k in range(segments_nb)]
s = sum(distribution)
distribution=[i/s for i in distribution]

bests=[]
bests_tempos=[]
max_reward=-100
min_reward=100

def random_minibatch(samples,size):
    if np.random.uniform()<0.3: 
#        print(bests)
        return np.random.choice(bests,size)
    np.random.shuffle(samples)
    return samples[:size]


segments_range =None
finished=False
def td_error_first_minibatch(samples,size,sort=True,segments_range=None, finished=False):
#    if np.random.uniform()<0.2: 
#        return np.random.choice(bests,size)
    if sort:
        samples = sorted(samples, reverse=True, key=lambda x:x.td_error)
##    segment_index= min(np.random.choice(range( size ),1, p=distribution)[0],size-2)
#    segment_index=0   
##    inf2 = segment_index*len(samples)/size
##    sup2 = min(1+(segment_index+1)*len(samples)/size, len(samples))
#    inf2 = segments_range[segment_index]
##    print(segment_index)
#    sup2 = min(segments_range[segment_index+1], batchsize-1)
##    print(inf2,sup2)
#
#    selection = np.random.choice(range(int(inf2), int(sup2) ),size)
#
#    return [samples[i] for i in selection]
        

    selections = []
    for i in range(size):
        if np.random.uniform()<0.1: 
            selections.append(bests[np.random.randint(len(bests))])
        elif np.random.uniform()<0.1: 
            selections.append(bests_tempos[np.random.randint(len(bests_tempos))])
        else:
            segment_index= min(np.random.choice(range( segments_nb ),1, p=distribution)[0],segments_nb-2)
        #    segment_index=0   
            inf2 = segments_range[segment_index]
            sup2 = min(segments_range[segment_index+1], batchsize-1)
            selections.append(sample_batch[np.random.randint(int(inf2), int(sup2))])
#    print (selections)
    return selections

def get_normal_reward(reward):
#    return reward
    if reward<0:
        return reward/min_reward*(-1)
#        return -1
    elif reward>0:
        return reward/max_reward*1
    else :
        return 0
#    if reward<0:
#        return reward/min_reward*(-1)
#    elif reward>0:
#        return reward/max_reward
#    else :
#        return 0
#    if max_reward-min_reward==0:
#        return max_reward
#    return (reward-min_reward)/(max_reward-min_reward)*2-1

def get_noised_action(action,noise=0.2):
    action = min(max(-1,action[0]),1)
    return [min(max(-1,action+np.random.uniform(-noise,noise)),1)]

import numpy as np
from tf_approach import *

"""
for i_episode in range(20):
    state = env.reset()
    for t in range(100):
        env.render() 
        action = env.action_space.sample()
        
        next_state, reward, done, info = env.step(action)
        smpl = Sample(s=state,a=action,ns=next_state,r=reward)
        print(smpl)
        
#        storeSamples(smpl)
        
        state=next_state
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
"""


def storeSamples(b):
#    if isinstance(b,list):
#        sample_batch.extend(b)
#    elif isinstance(b,Sample):
    sample_batch.append(b)
#    print(b.reward, max_reward)
    bests_tempos
    while len(bests)>10000:
        bests.pop(0) 
    while len(bests_tempos)>5000:
        bests_tempos.pop(0)
    while len(sample_batch)>batchsize:
        sample_batch.pop(np.random.randint(10000, batchsize-1))
#
#
#print("warmup starts")
#
#        
#
#print("max_reward: "+str(max_reward))

min_episode =10000
success=0
total_step=0

nb_experience=10
    
for i_exp in range(nb_experience):
    print("###DDPG ",i_exp,"starts")
    for i_episode in range(300):
    #    print("Episode "+str(i_episode)+":")
    #    print("max_reward: "+str(max_reward))
    #    print("min_reward: "+str(min_reward))
        state = env.reset()
        state=np.reshape(state,newshape=([2]))
        tempo = []
        reward_sum=0
#        rewardlist=[]
        for t in range(999):
            if i_episode <100 and success <5:
                action = env.action_space.sample()*100
            else:
                action = get_action(state)
            action=np.reshape(action,newshape=([1]))
            action = get_noised_action(action)
    
            next_state, reward, done, info = env.step(action)
                
            next_state=np.reshape(next_state,newshape=([2]))
            reward_sum+=reward
#            rewardlist.append(reward)
    
            sample = Sample(s=state,a=action,ns=next_state,r=reward)
            tempo.append(sample)
            if(done and t<998 ):
                success+=1
                bests.append(sample)
            if (done and t<998 and t<min_episode+100):
                min_episode=(t+1)
                bests_tempos.extend(tempo)
            if max_reward<reward:
                max_reward=reward
            if min_reward>reward:
                min_reward=reward 
            storeSamples(sample)
            state=next_state
    #        samples =  random_minibatch(sample_batch,minibatch_size)
            if t%20==0:
                sort=True
            else:
                sort=False 
            if not(i_episode <100 and success <5):
                if segments_range==None:
                    segments_range = [int(s*len(sample_batch)/segments_nb) for s in range(segments_nb)]
                elif len(sample_batch)<batchsize and len(sample_batch)%10000 <1500:
                    segments_range = [int(s*len(sample_batch)/segments_nb) for s in range(segments_nb)]
                elif len(sample_batch)==batchsize and not finished:
                    segments_range = [int(s*batchsize/segments_nb) for s in range(segments_nb)]
                    
                samples = td_error_first_minibatch(sample_batch,minibatch_size,sort,segments_range,finished)
                if len(sample_batch)==batchsize:
                    finished=True
                    
    #            samples = random_minibatch(sample_batch,minibatch_size)                
                
                next_state_samples=[]
                state_samples=[]
                action_samples=[]
                reward_samples=[]
                for s in samples:  
                    next_state_samples.append(s.next_state)
                    action_samples.append(s.action)
                    state_samples.append(s.state)
                    reward_samples.append([get_normal_reward(s.reward)])
                    
                loss_value = training(state_samples, action_samples, next_state_samples,reward_samples,samples)
            
#                if i_episode%5==0:
#                    env.render() 
            else:
                update_tderror(sample,get_normal_reward(sample.reward))  
                
    #        print(sample.td_error)   
    #        print(sample.critic)   
    #        print(sample.next_critic)
    #        print(sample)
            if done and t+1<999:
        #            print("**************Episode finished after "+str(t+1)+" timesteps***************")
                break
        total_step+=(t+1)
        print('episode ',i_episode,'nb steps ',t+1, " perf : ", int(reward_sum) , " total steps :", total_step)
     
#
        if not(i_episode <100 and success <5):
    #        for i in range(20):
    #            print(sample_batch[i])
            samples = sorted(samples,reverse=True,key=lambda x:x.critic[0])
#            samples = sorted(samples,reverse=True,key=lambda x:x.state[0])
#            for i in range(40):
#                print(samples[i])
#            print(loss_value, 'success',success)
            
#            set_training_parameters(clr=max(0.00001, critic_learning_rate*0.99**i_episode))
            decent_training_parameters(rate=0.98,min_var=0.00001)

   
    sess.run(init)
    sample_batch = []
    bests=[]
    bests_tempos=[]
    min_episode =10000
    success=0
    total_step=0
    
    segments_range =None
    finished=False
    max_reward=-100
    min_reward=100
    set_training_parameters(critic_learning_rate) 
        
        
             
        
        
        
        