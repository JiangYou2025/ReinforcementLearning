# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 12:20:33 2019

@author: you
"""
import tfplot
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.losses.losses_impl import Reduction
from sumaryutils import HeatMap

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).""" 
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean) 
    
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def state_summary(states, axis=0):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization).""" 
  tf.summary.histogram('States', states[:, axis])
def state_critic_summary(state_critics):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization).""" 
  tf.summary.histogram('Critic_On_States', state_critics)
  
class RLModel:
    def __init__(self, model, net_params={}, env_params={}):
        self.state_size = 2
        self.action_size = 1
        self.critic_size = 1
        self.reward_size = 1
        
        self.hidden_layer_size = 50
        self.hidden_layer_nbr = 2
        self.critic_learning_rate = 0.0003
        self.critic_decent_rate = 0.95
        self.min_c_learning_rate = 0.000001
        
        self.actor_learning_rate = 0.001
        self.actor_decent_rate = 0.95
        self.min_a_learning_rate = 0.000001
        
        self.gamma = 0.99
        
        self.use_summary = False
        self.summary_path = 'summaries'
        
        # Update Parameters
        for k, v in net_params.items():
            setattr(self, k, v)
            print("update", k, "to", v)
    
        for k, v in env_params.items():
            setattr(self, k, v)
            print("update", k, "to", v)
            
        self.state_layer = None
        self.action_layer = None
        self.next_state_layer = None
        self.reward_layer = None
        
        self.actor = None
        self.critic = None
        
        self.next_action_layer = None
        self.next_critic_layer = None
        
        self.heatmap = HeatMap()
        
        self.current_steps = None
        self.current_rewards = None
        
        self.critic_optimizer = None
        self.actor_updater = None
        
        self.loss = None
        
        self.sess = None
        
        self.train = None
        
        self.c_learning_rate_var = None
        self._init_place_holder()
        self._init_model(name=model)
        self.init_session()  
        
        self.merged_summary = tf.summary.merge_all()  
        self.summary_writer = tf.summary.FileWriter(self.summary_path+'/train',self.sess.graph)     
        with self.sess.as_default():                              
            tf.global_variables_initializer().run()
        
    def _init_place_holder(self):  
        
        # Update Parameters
         
        self.state_layer = tf.placeholder(dtype = tf.float32,shape=[None, self.state_size])#s,a
        self.action_layer = tf.placeholder(dtype = tf.float32,shape=[None, self.action_size])#s,a
        self.reward_layer = tf.placeholder(dtype = tf.float32,shape=[None, self.reward_size])#r

            
        self.next_state_layer = tf.placeholder(dtype=tf.float32,shape=[None, self.state_size])
        self.next_action_layer =  tf.placeholder(dtype=tf.float32,shape=[None, self.action_size])
        self.next_critic_layer = tf.placeholder(dtype=tf.float32,shape=[None, self.critic_size])
        
        self.current_steps = tf.placeholder(dtype=tf.float32,shape=[])
        self.current_rewards = tf.placeholder(dtype=tf.float32,shape=[])
        self.histo_state_critic = tf.placeholder(dtype=tf.float32,shape=[None,])
#        
#        self.critic_on_states = tf.placeholder(dtype=tf.image, shape=[])
        self.current_state_critics = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size + self.critic_size]) 
        self.current_state_actions = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size + self.action_size]) 
        
        summary_heatmap = tfplot.summary.wrap(self.get_state_critic_heatmap, batch=False)

        summary_heatmap("State Critic Heatmap", self.current_state_critics)
        summary_heatmap("State Action Heatmap", self.current_state_actions)
        
        with tf.name_scope('Performance'):
            tf.summary.scalar("Steps", self.current_steps)
            tf.summary.scalar('Rewards', self.current_rewards)
        state_summary(self.state_layer, axis=0)
        
#        tf.summary.text('Critic_On_States_Text',  self.critic_on_states)
#        state_critic_summary(self.histo_state_critic)
        
    
    def _init_model(self, name='basic'):
        if name == 'basic':
            self.actor, self.critic = self._create_basic_model(  self.state_layer, 
                                                                self.action_layer, 
                                                                self.next_state_layer, 
                                                                self.reward_layer, 
                                                                self.next_action_layer,
                                                                self.next_critic_layer,
                                                                self.hidden_layer_size,
                                                                self.hidden_layer_nbr
                                                                )
        elif name == 'resnet':
            self.actor, self.critic = self._create_combined_model(  self.state_layer, 
                                                                self.action_layer, 
                                                                self.next_state_layer, 
                                                                self.reward_layer, 
                                                                self.next_action_layer,
                                                                self.next_critic_layer,
                                                                self.hidden_layer_size,
                                                                self.hidden_layer_nbr
                                                                )
        self._init_loss() 
     
        
    def _init_loss(self):
        #critic_pred = tf.minimum(1000.0,tf.maximum(-1000.0,next_critic*gamma +reward_layer))#critic 
        critic_pred = self.next_critic_layer*self.gamma + self.reward_layer#critic 
#        critic_ = tf.maximum(-100.0, tf.minimum(200.0, self.critic))
#        critic_pred_ = tf.maximum(-1.0, tf.minimum(100.0,critic_pred))
        critic_ =  self.critic 
        critic_pred_ =  critic_pred  
        with tf.name_scope('Critic'):
            variable_summaries(critic_)
        with tf.name_scope('Critic_pred'):
            variable_summaries(critic_pred)
        
        #Define  Loss
        #loss = tf.losses.mean_squared_error(labels=critic_pred_, predictions=critic_ ,reduction=Reduction.NONE)
        self.loss = tf.losses.mean_squared_error(labels=critic_pred_, predictions=critic_ ,reduction=Reduction.NONE)
        
        with tf.name_scope('Loss'):
            variable_summaries(self.loss)
        
        #Define  Critic Optimizer
        self.c_learning_rate_var = tf.Variable(self.critic_learning_rate, trainable=False)
        optimizer = tf.train.AdamOptimizer(self.c_learning_rate_var)
        self.train = optimizer.minimize(self.loss)
         
        #Define  Actor Parameters
        params=[]
        with tf.variable_scope('actor', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('kernel', shape=[self.hidden_layer_size, self.action_size])
            b = tf.get_variable('bias', [self.action_size])
#            variable_summaries(w)
#            variable_summaries(b)
            params.extend([w,b])
        for i in range(self.hidden_layer_nbr):
            with tf.variable_scope('ah'+str(i), reuse=tf.AUTO_REUSE):
                w = tf.get_variable('kernel', [None, self.hidden_layer_size])
                b = tf.get_variable('bias', [self.hidden_layer_size])
#                variable_summaries(w)
#                variable_summaries(b)
                params.extend([w,b])
#            with tf.variable_scope('an'+str(i), reuse=tf.AUTO_REUSE):
#                w = tf.get_variable('kernel', [1, self.hidden_layer_size])
#                b = tf.get_variable('bias', [self.hidden_layer_size])
#                variable_summaries(w)
#                variable_summaries(b)
#                params.extend([w,b])
            
        #Define  Actor Updater
        grad_v = tf.gradients(self.critic, self.action_layer)
        #print(grad_v)
        
        grad = grad_v[0]/tf.to_float(tf.shape(grad_v[0])[0])
        #print(grad)
        params_grad = tf.gradients(self.actor, params, -grad)
        #print(params_grad)
        adam = tf.train.AdamOptimizer(self.actor_learning_rate)
        self.actor_updater = adam.apply_gradients(zip(params_grad, params))
         
        
    def init_session(self):
        init = tf.global_variables_initializer() 
        if self.sess:
            self.sess.close()
        self.sess = tf.Session()
        self.sess.run(init)
        self.reset_training_rate(self.critic_learning_rate)
         
        
    def _create_combined_model(self, state_layer, 
                                 action_layer, 
                                 next_state_layer, 
                                 reward_layer, 
                                 next_action_layer,
                                 next_critic_layer, 
                                 hidden_layer_size,
                                 hidden_layer_nbr):
        #linear_model = tf.layers.Dense(units=1)
        hd_s = tf.layers.dense(state_layer, units=hidden_layer_size, activation=tf.nn.relu)
        hd_a1 = tf.layers.dense(tf.concat([hd_s,state_layer],1), units=hidden_layer_size, activation=tf.tanh,name='ah0')
        #hd_a2 = tf.layers.dense(tf.concat([hd_a1,state_layer],1),units=hiddenlayer_size,activation=tf.nn.softplus,name='a2')
        
        actor = tf.layers.dense(hd_a1, units=1, name='actor', activation=tf.tanh)#it will take the next state as inputs
         
        hd_sa1 = tf.layers.dense(tf.concat([hd_s,self.action_layer],1),units=hidden_layer_size,activation=tf.nn.relu)
        hd_sa1bn = tf.layers.batch_normalization(hd_sa1)
        hd_sa2 = tf.layers.dense(tf.concat([hd_sa1bn,state_layer, action_layer],1), units=hidden_layer_size,activation=tf.nn.relu)
        hd_sa2bn = tf.layers.batch_normalization(hd_sa2)
        hd_sa3 = tf.layers.dense(tf.concat([hd_sa2bn,state_layer, action_layer],1), units=hidden_layer_size,activation=tf.tanh)
        critic =  tf.layers.dense(hd_sa3, units=1,name='critic',activation=tf.nn.softplus)
        return actor, critic
        
    def _create_basic_model(self, state_layer, 
                                 action_layer, 
                                 next_state_layer, 
                                 reward_layer, 
                                 next_action_layer,
                                 next_critic_layer, 
                                 hidden_layer_size,
                                 hidden_layer_nbr):
        hd_bn = None
        for i in range(hidden_layer_nbr):
            if hd_bn is None:
                hd_s = tf.layers.dense(state_layer, units=hidden_layer_size, name='ah'+str(i), activation=tf.nn.relu)
            else:
                hd_s = tf.layers.dense(hd_bn, units=hidden_layer_size, name='ah'+str(i), activation=tf.nn.relu)
            hd_bn = tf.layers.batch_normalization(hd_s, name='an'+str(i))
        actor = tf.layers.dense(hd_bn, units=1, name='actor', activation=tf.tanh)#it will take the next state as inputs
        
        hd_bn = None
        for i in range(hidden_layer_nbr):
            if hd_bn is None:
                hd_sa = tf.layers.dense(tf.concat([state_layer, action_layer], 1), units=hidden_layer_size, activation=tf.nn.leaky_relu)
            else:
                hd_sa = tf.layers.dense(hd_bn, units=hidden_layer_size, activation=tf.nn.leaky_relu)
            hd_bn = tf.layers.batch_normalization(hd_sa)
        critic = tf.layers.dense(hd_bn, units=1, name='critic', activation=tf.nn.softplus)
        return actor, critic
        
    def get_action(self, state):
        return self.sess.run(self.actor,feed_dict={self.state_layer: [state]})
    
    
    def training(self, state_samples, action_samples,next_state_samples,reward_samples,samples):
            
        next_action_value = self.sess.run(self.actor,feed_dict={self.state_layer: next_state_samples})
        next_critic_values = self.sess.run(self.critic,feed_dict={self.state_layer:next_state_samples,
                                                        self.action_layer:next_action_value})
        critic_value = self.sess.run(self.critic,feed_dict={self.state_layer:state_samples,
                                                  self.action_layer:action_samples})
    
    #    print (inputs_layer_samples)
    #    print (reward_layer_samples)
    #    print (next_critic_value)
    #    print (next_inputs_layer_samples)
        
        _, losses = self.sess.run((self.train, self.loss),feed_dict={self.state_layer: state_samples,
                                                  self.action_layer:action_samples, 
                                                 self.reward_layer: reward_samples,   
                                                 self.next_critic_layer:next_critic_values})
        loss_value = np.sum(losses)
    #    print (losses)
        for i in range(len(samples)):
            samples[i].td_error = losses[i]
            samples[i].next_critic = next_critic_values[i]
            samples[i].critic = critic_value[i]
            samples[i].ma_td_err = samples[i].ma_td_err*0.99 + losses[i]*0.01
            
                
        action_value = self.sess.run(self.actor,feed_dict={self.state_layer: state_samples})
    #    critic_value = sess.run(critic,feed_dict={state_layer:state_samples,
    #                                          action_layer:action_value})
        self.sess.run(self.actor_updater, feed_dict={self.state_layer:state_samples,
                                    self.action_layer:action_value})
    
        
    
        return loss_value

    def get_td_error(self, action, state, next_state, normal_reward):
        
        next_action_value = self.sess.run(self.actor,feed_dict={self.state_layer: [next_state]})
        next_critic_value = self.sess.run(self.critic,feed_dict={self.state_layer: [next_state],
                                                        self.action_layer:next_action_value})
        
        critic_value = self.sess.run(self.critic,feed_dict={self.state_layer:[state],
                                                  self.action_layer:[action]})
        
        td_err = self.sess.run(self.loss,feed_dict={self.state_layer: [state],
                                                    self.action_layer: [action], 
                                                    self.reward_layer: [[normal_reward]],   
                                                    self.next_critic_layer: next_critic_value})
    
        return critic_value[0, 0], next_critic_value[0], td_err[0]
        
    def reset_training_rate(self, clr=None):
        if clr!=None:
            self.sess.run(self.c_learning_rate_var.assign(clr))
            
    def get_critic_learning_rate(self):
        return self.sess.run(self.c_learning_rate_var)
    
    def decent_training_rate(self): 
        self.sess.run(self.c_learning_rate_var.assign(tf.maximum(self.min_c_learning_rate,  self.c_learning_rate_var*self.critic_decent_rate)))
            
    
    def write_summaries(self, state_samples, action_samples,next_state_samples,reward_samples, i_episode, current_rewards, current_steps): 
        next_action_value = self.sess.run(self.actor, feed_dict={self.state_layer: next_state_samples})
        next_critic_values = self.sess.run(self.critic, feed_dict={self.state_layer: next_state_samples,
                                                        self.action_layer:next_action_value}) 
    
        action_values = self.sess.run(self.actor, feed_dict={self.state_layer: state_samples})
        critic_values = self.sess.run(self.critic, feed_dict={self.state_layer: state_samples,
                                                        self.action_layer:action_values})  
    
#        state_critic, texts = self.get_histo_state_critic(state_samples, critic_values)
        
        state_critics = [[np.round(state_samples[i][0], 1), np.round(state_samples[i][1], 2),np.round(critic_values[i], 2)] for i in range(len(state_samples))]   
        state_actions = [[np.round(state_samples[i][0], 1), np.round(state_samples[i][1], 2),np.round(action_values[i], 2)] for i in range(len(state_samples))]   
        
        summary, _ = self.sess.run([self.merged_summary, self.loss], 
                                   feed_dict={self.state_layer: state_samples,
                                              self.action_layer:action_samples, 
                                              self.reward_layer: reward_samples,   
                                              self.next_critic_layer:next_critic_values,
                                              self.current_steps: current_steps,
                                              self.current_rewards:current_rewards,
#                                              self.histo_state_critic:state_critic,
#                                              self.critic_on_states:texts,
                                              self.current_state_critics:state_critics, 
                                              self.current_state_actions:state_actions, 
                                             })
        
        self.summary_writer.add_summary(summary, i_episode)
        self.summary_writer.flush()
#    
#    
#    def get_histo_state_critic(self, state_samples, critic_values):
#        states_map = dict()
#        for pos, c in list(zip([p for p,v in state_samples], critic_values)):
#            s = np.round(pos, 2)
#            if s in states_map:
#                if states_map[s] < c:
#                    states_map[s] = c
#            else:
#                states_map.update({s:c})
#        sum_critic = sum(states_map.values()) 
#        result = []
#        for s, c in states_map.items():
#            count = int(c/sum_critic*10000)
#            result.extend([s]*count) 
#            
#        ls = sorted([[s, np.round(c[0], 2)] for s, c in states_map.items()], key=lambda x:x[1], reverse=True)
#        return result, str(ls) 
        

    
    def get_state_critic_heatmap(self, data):
        mat = np.zeros(shape=[15, 18])
        x = np.linspace(-0.07, 0.07, 15)
        y = np.linspace(-1.2, 0.5, 18)
        color = 'YlGn'
        for i in range(data.shape[0]): 
            pos, velo, value = data[i]  
            #        print(i, pos, velo, critic)
            idx, idy = int(round((velo + 0.07) *100, 0)), int(round((pos + 1.2) * 10, 0))
            if abs(mat[idx, idy]) < abs(value):
                mat[idx, idy] = value
            if value < 0 :
                color = "RdYlGn"
        fig = self.heatmap.get_heat_map(mat, np.round(x,2), np.round(y,2), color=color)
        return fig