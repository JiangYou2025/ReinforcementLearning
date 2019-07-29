# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 12:02:05 2018

@author: you
"""

import numpy as np

def read_mat(file_name='reslutat_50.txt'):
    f=open(file_name)
    mat=[]
    
    for line in f.readlines():
        if '#' in line or'WARN' in line:
            continue
        else:
            line = line.replace('\n','')
            a = [int(s) for s in line.split(' ') if s.lstrip('-+').isdigit()]
#            print(a)
            mat.append(np.array(a))
    return np.array(mat)

#mat = read_mat(file_name='reslutat_50.txt')
mat = read_mat(file_name='resultat_100.txt')
#mat.shape

import matplotlib.pyplot as plt

def get_average_mat(mat,nb_experiences=10,nb_steps=300):
    mat_sort = sorted(mat,key = lambda x:x[3])
    new_mat=[]
    for i in range(nb_steps):
        new_mat.append(np.mean(mat_sort[i*nb_experiences:(i+1)*nb_experiences],axis=0))
    return np.array(new_mat)
      
new_mat = get_average_mat(mat,nb_experiences=10,nb_steps=300)

def plot_mat(new_mat,title_size='50'):
    plt.figure(1)       
    x = new_mat[:,3]
    y2= new_mat[:,2]
    plt.plot(x,y2)
    plt.ylabel('Performence')
    plt.xlabel('Total Steps')
    plt.title('Experience with '+title_size+' Nodes each layer')
    plt.show()
        
    plt.figure(2)       
    x = new_mat[:,3]
    y2= new_mat[:,1]
    plt.plot(x,y2)
    plt.title('Experience with '+title_size+' Nodes each layer')
    plt.ylabel('Steps Per Episodes')
    plt.xlabel('Total Steps')
    plt.show()
       
#plot_mat(new_mat,title_size='50')
plot_mat(new_mat,title_size='100')
        
        
        


        
        
        
        
