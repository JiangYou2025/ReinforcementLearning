# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:40:58 2019

@author: you
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

def read_mat(file_name='reslutat_50.txt'):
    f=open(file_name)
    mat=[]
    
    for line in f.readlines():
        if 'Episode' not in line:
            continue
        else:
            line = line.replace('\n','')
            line = line.replace('[',' ')
            line = line.replace(']',' ')
            a = [int(s) for s in line.split(' ') if s.lstrip('-+').isdigit()]
#            print(a)
            mat.append(np.array(a))
    mat = np.array(mat)
    mat = mat[:, [0,1,2,-1]]
    return mat

#mat.shape

def get_average_mat(mat,nb_experiences=10,nb_steps=300):
    mat_sort = sorted(mat,key = lambda x:x[3])
    new_mat=[]
    for i in range(nb_steps):
        new_mat.append(np.mean(mat_sort[i*nb_experiences:(i+1)*nb_experiences],axis=0))
    return np.array(new_mat)
      

def plot_mat(new_mat,title='50'):
    plt.figure(1)       
    x = new_mat[:,3]
    y2= new_mat[:,2]
    plt.plot(x,y2)
    plt.ylabel('Performence')
    plt.xlabel('Total Steps')
    plt.title('Experience with '+title)
#    plt.show()
    plt.savefig('Experience_'+title+'_Performence.png')
        
    plt.figure(2)       
    x = new_mat[:,3]
    y2= new_mat[:,1]
    plt.plot(x,y2)
    plt.title('Experience with '+title)
    plt.ylabel('Steps Per Episodes')
    plt.xlabel('Total Steps')
#    plt.show()
    plt.savefig('Experience_'+title+'_Steps_Per_Episodes.png')
       
        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("python print_result.py [txt_file_name] [title]")
        exit()
    f_name = sys.argv[1]
    title = sys.argv[2]
    #mat = read_mat(file_name='reslutat_50.txt')
    mat = read_mat(file_name=f_name)
    new_mat = get_average_mat(mat,nb_experiences=10,nb_steps=300)
    #plot_mat(new_mat,title_size='50')
    plot_mat(new_mat,title=title)
    
