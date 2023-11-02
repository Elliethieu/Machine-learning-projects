import gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict

'''instruction:
https://towardsdatascience.com/q-learning-for-beginners-2837b777741

and the packages we need are gym, pygame, torch, torchaudio, torchvision'''

EPISODES =  20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":

    random.seed(1)
    n=np.random.seed(1)
    env = gym.envs.make("FrozenLake-v1")
    #env.seed(1)
    env.action_space.n


    # need to update the Q_table in iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        #(state, prop) = env.reset()
        state = env.reset() #this is for the older version that they test

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with Q-Learning

        while (not done):
            rnd = np.random.random() #generate a random number between 0 and 1
            if rnd < EPSILON:
                action = env.action_space.sample() # performs a random action.

            else:
                extract_dict = defaultdict(default_Q_value)
                for j in range(4):
                    extract_dict[(state, j)] = Q_table[(state, j)]
                (state,action) = max(extract_dict, key=extract_dict.get)

            #new_state, reward, done, info, prop = env.step(action)
            new_state, reward, done, info = env.step(action) #again, this is for the older version
            episode_reward += reward  # update episode reward

            if done == False:
                Q_values_new_state = [Q_table[new_state,k] for k in range(4)]
                a = max(Q_values_new_state)
                Q_table[(state, action)] = (1-LEARNING_RATE)*Q_table[(state, action)] + \
                                    LEARNING_RATE * (reward + DISCOUNT_FACTOR *a )
            else:
                Q_table[(state, action)] = (1 - LEARNING_RATE) * Q_table[(state, action)] + \
                                           LEARNING_RATE * reward

            state = new_state #update state

        EPSILON = EPSILON* EPSILON_DECAY #update epsilon after a succesful episode

        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward)

        if i%100 == 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################