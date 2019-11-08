from __future__ import division
import gym,random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam  
import numpy as np
import pandas as pd
import os
import librosa as li
import numpy as np
class DQNAgent():
    def __init__(self, env_id, path, episodes, max_env_steps, win_threshold, epsilon_decay,
                 state_size=None, action_size=None, epsilon=1.0, epsilon_min=0.01, 
                 gamma=1, alpha=.01, alpha_decay=.01, batch_size=16, prints=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make(env_id)
        if state_size is None: 
            self.state_size = self.env.observation_space.n 
        else: 
            self.state_size = state_size
 
        if action_size is None: 
            self.action_size = self.env.action_space.n 
        else: 
            self.action_size = action_size
        self.episodes = episodes
        self.env._max_episode_steps = max_env_steps
        self.win_threshold = win_threshold
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        self.path = path                     
        self.prints = prints                 
        self.model = self._build_model()
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        
        return model
    def act(self, state):
        if (np.random.random() <= self.epsilon):
            return self.env.action_space.sample()
        return self.model.predict(state)
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def replay(self, x,y):
        x_batch, y_batch = [], []
        x_batch=x
        y_batch=y
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
if __name__=="__main__":
    DQN=DQNAgent("CartPole-v1","Actor_01/",100,50,0.7,0.56,state_size=4,action_size=1)
    path=r"E:\\python\\Actor_"
    zcr=[]
    rms=[]
    mfc=[]
    spc=[]
    tar=[]
    for i in range(1,25):
        if(i<10):
            path+="0"+str(i)
        else:
            path+=str(i)
        for r,d,f in os.walk(path):
            for file in f:
                at,sr=li.load(path+"\\"+file)
                zcr.append(np.mean(li.feature.zero_crossing_rate(at)[0]))
                rms.append(np.mean(li.feature.rms(at)[0]))
                mfc.append(np.mean(li.feature.mfcc(at)[0]))
                spc.append(np.mean(li.feature.spectral_centroid(y=at,sr=sr)[0]))
                tar.append(file.split("-")[2])
    ftur=[zcr,rms,mfc,spc]
    res=[tar]
    f=[[] for x in range(len(zcr))]
    for j in range(len(zcr)):
        f[j].append([zcr[j],rms[j],mfc[j],spc[j]])
    f=np.array(f)
    f=np.reshape(f,(60,4))
    DQN._build_model()
    a=np.array(res)
    res=np.reshape(a,(4,15))
    DQN.replay(f[:14],res[0][:-1])
    print(DQN.act(f[:-5]))                             
