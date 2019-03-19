# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# import matplotlib.pyplot as plt
import pandas as pd 
import random
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class environment:
    def __init__(self, stock_1 ,stock_2 ,capital ,window_size):
        self.action_space = [0.25,0.1,0.05,0,0.05,0.1,0.25]
        self.stocks = [stock_1,stock_2]
        self.capital = capital
        # Create feature of both stock
        self.feature, self.close_price, self.open_price = self.create_feature(window_size,stock_1,stock_2)
        # Get the id of last day in data
        self.last_day_number = self.close_price.shape[0]-1
        ### Build the variable
        ### *** Note that the value will be assigned in reset function
        ###    - day id 
        self.day = 0
        ###    - Random initial portfolio
        self.initial_portfolio = 0
        ###    - Create the number of share of each stock
        self.amount_share = 0
        ###    - Current portfolio value
        self.current_portfolio_value = 0
        ###    - done
        self.done = False
        
    def reset(self):
        self.day = 0#random.randint(0,self.last_day_number)
        self.done = False
        self.initial_portfolio = np.random.dirichlet(np.ones(2))
        self.amount_share = self.capital*self.initial_portfolio/self.open_price[self.day]
        self.current_portfolio_value = self.cal_portfolio_value_at_the_end_of_day()
        #### Create state 
        state = np.hstack((self.feature[self.day],self.current_portfolio_value/np.sum(self.current_portfolio_value)))
        return state
        
    def step(self,action_id):
        action = self.action_space[action_id]
        ### Calculate the portfolio value according to the action
        if action_id<3:
            ### If action_id <3, then sell x% of values of stock 1 and buy the corresponding amount of stock2 
            adjust_amount = self.current_portfolio_value[0]*action
            new_portfolio_fraction = [ self.current_portfolio_value[0] - adjust_amount\
                                      ,self.current_portfolio_value[1] + adjust_amount]
        elif action_id>3:
            ### If action_id >3, then sell x% of values of stock 2 and buy the corresponding amount of stock1 
            adjust_amount = self.current_portfolio_value[1]*action
            new_portfolio_fraction = [ self.current_portfolio_value[0] + adjust_amount\
                                      ,self.current_portfolio_value[1] - adjust_amount]
        else:
            ### If action_id = 3, then do nothing
            new_portfolio_fraction = self.current_portfolio_value
            
        ### Shift to next day
        self.day += 1
        ###    - if the final day in data have been reached, then done = true
        if self.day == self.last_day_number: self.done = True
            
        ### Update portfolio (buy at open price of the day)
        ###      **** Note that the open price is the close price of the previous day
        self.amount_share = new_portfolio_fraction/self.open_price[self.day]
        
        ### Calculate reward
        ###    - Store old portfolio value
# #         print self.current_portfolio_value
        total_old_portfolio_value = np.sum(self.current_portfolio_value)
        ###    - Calculate new portfolio value
        self.current_portfolio_value = self.cal_portfolio_value_at_the_end_of_day()
#         print self.current_portfolio_value
        ###    - Calculate reward
        total_current_portfolio_value = np.sum(self.current_portfolio_value)
        reward =  100.0*(total_current_portfolio_value-total_old_portfolio_value)/total_old_portfolio_value
        
        ### Create state
        ###    - stock feature
        stock_feature = self.feature[self.day]
        ###    - proportion feature
        proportion_feature = self.current_portfolio_value/np.sum(self.current_portfolio_value)
        ###    - merge features
        next_state = np.hstack((self.feature[self.day],proportion_feature))
        
        return next_state,reward,self.done
    
    def cal_portfolio_value_at_the_end_of_day(self):
        return self.amount_share*self.close_price[self.day]

    def feature_of_stock(self, stock, window_size):
        ### Read data and select only close price and sample only 3300 day for being train set
        price = pd.read_csv(stock)[['Close']][:3300]
        ### Lag the close price for (window_size+1) time
        for i in range(window_size+1):
            price['lag{}'.format(i+1)] = price['Close'].shift(i+1)
        ### Create percent change between each consecutive day (0-100%)
        for i in range(window_size):
            price['percent_change{}'.format(i+1)] = (price['lag{}'.format(i+1)]-price['lag{}'.format(i+2)])*100\
                                                    /price['lag{}'.format(i+2)]
        price = price.dropna()
        ### Select only the percent change as the feature
        feature = np.array(price[['percent_change{}'.format(i+1) for i in range(window_size)]].values.tolist())
        ### Return feature as feature, 
        ###        price['close'] as the sell price at that day,
        ###        price[lag] as a buy price at that day
        return feature,price['Close'],price['lag1']
    
    def create_feature(self, window_size = 7, stock1 = 'high_volatile/APA.csv', stock2 = 'low_volatile/JNJ.csv'):
        ### Generate feature of each stock
        feature_1,close_price_1,open_price_1 = self.feature_of_stock(stock1,window_size)
        feature_2,close_price_2,open_price_2 = self.feature_of_stock(stock2,window_size)
        ### Concat feature from 2 stocks to be one features
        feature = np.concatenate([feature_1,feature_2],axis=1)
        ### Concat buy price and sell price to format [[<buyorsell>ofstock1,<buyorsell>ofstock2], [],[]]
        ### example of buy price
        ###                 [[buy price of stock 1 in day 0, buy price of stock 2 in day 0]
        ###                  [buy price of stock 1 in day 1, buy price of stock 2 in day 2],....]
        close_price = np.stack((close_price_1,close_price_2),axis=-1)
        open_price = np.stack((open_price_1,open_price_2),axis=-1)
        return feature,close_price,open_price
    
    
# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.15 #0.01 in test1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        ### First hidden layer
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        ### Second hidden layer
        model.add(Dense(24, activation='relu'))
        ### Output layer
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            try:
                target_f[0][action] = target
            except:
                print "Target {}".format(target)
                raise ValueError('A very specific bad thing happened.')
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        
        
if __name__ == "__main__":
    window_size = 7
    capital = 100000000.0,
    env = environment('high_volatile/APA.csv','low_volatile/JNJ.csv',capital,window_size)
    print "#############"
    print "Number of Training Day {}".format(env.close_price.shape[0])
    print "#############"
    state_size = window_size*2 + 2
    action_size = 7
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 16 #32 in test1
    print "Save model at episode 0"
    agent.model.save('model_test_5/{}.h5'.format(0)) 
    print "Start to train"
    for e in range(EPISODES):
        print "####################"
        print "Episode {}".format(e+1)
        print "####################"
        state = env.reset()
        print "start at day {}".format(env.day)
        state = np.reshape(state, [1, state_size])
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if (e)%10 == 0:
            print "Save model at episode {}".format(e+1)
            agent.model.save('model_test_5/{}.h5'.format(e+1)) 