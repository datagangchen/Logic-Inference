from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import random

def load_data(train_data):
    data = loadmat(train_data)
    trajs = data['trajs']
    label = data['label']
    name  = data['name']
    namelist = name.tolist()
    time1 = trajs[0]['time']
    label = np.squeeze(label)
    sigsets =[]
    for index in range(trajs.size):
        sigsets.append(trajs[index]['X'][0])
    return sigsets, time1[0][0], namelist, label

def generate_vocab(num,namelength,tlow,thigh,slow,shigh):
    vocab =[]
    word =[]
    intervel =[]
    intervel.append(round(random.uniform(tlow,thigh),2))
    intervel.append(round(random.uniform(tlow,thigh),2))
    intervel.sort()
    intervel2 =[]
    intervel2.append(round(random.uniform(tlow,thigh),2))
    intervel2.append(round(random.uniform(tlow,thigh),2))
    intervel2.sort()    
    for i in range(num):
        word =[]
        if i< int(0.2*num):
            word.append(0)
            word.append(random.randint(1,4))
            word.append(random.randint(0,namelength-1))
            word.append(random.randint(1,2))
            word.append(round(random.uniform(slow,shigh),3))
            if word[1] ==1 or word[1]==2:
                word.extend(intervel)
                word.extend([0,0])
            else:
                word.extend(intervel)
                word.extend(intervel2)
        else:
            word.append(random.randint(1,2))
            word.append(random.randint(1,4))
            word.append(random.randint(0,namelength-1))
            word.append(random.randint(1,2))
            word.append(round(random.uniform(slow,shigh),3))
            if word[1] ==1 or word[1]==2:
                word.extend(intervel)
                word.extend([0,0])
            else:
                word.extend(intervel)
                word.extend(intervel2)      

        vocab.append(word)           

    return  vocab                 








