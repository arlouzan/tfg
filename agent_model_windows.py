import os
import cv2
import PIL.Image
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
import keras.backend as K
from collections import deque
import time
import random
from tqdm import tqdm
import numpy as np
import json
import datetime


base_dir =os.getcwd()
target_dir = os.path.join(base_dir,"dataset","training","images_selected","game_1")
annotation_dir=os.path.join(base_dir,"dataset","training","annotations","game_1")


target_directory = target_dir

annotation_directory = os.path.join(annotation_dir,"ball_markup.json")

VIDEO_RESOLUTION_X = 1920
VIDEO_RESOLUTION_Y = 1080

OBSERVATION_SPACE_VALUES = (200,400,1)
ACTION_SPACE_SIZE = 5 #left, stand still, right, up, down

def annotation_list(directory):
	json_file = open(directory)
	aux_list = []
	distros_dict = json.load(json_file)
	for key,value in distros_dict.items():
		aux_list.append(value)
	return aux_list


def _rgb_to_grayscale(image):
    """
    Convert an RGB-image into gray-scale using a formula from Wikipedia:
    https://en.wikipedia.org/wiki/Grayscale
    """

    # Get the separate colour-channels.
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Convert to gray-scale using the Wikipedia formula.
    img_gray = 0.2990 * r + 0.5870 * g + 0.1140 * b
	
    return img_gray



DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 200  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 40  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 20  # Terminal states (end of episodes)

MIN_REWARD = 10000000 # For model save


# Environment settings
EPISODES = 500

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 1000  # episodes


def FindPoint(x1, x2, y1, y2, x, y):
    return (x > x1 and x < x2 and y > y1 and y < y2) 

  
class Enviroment:

	OFF_CENTER_PENALTY = 50
	PLAYER_MOVE = 10
	ACTION_SPACE_SIZE = 5 # Move left, stay still, move right
	LEFT = 0
	NO_OP = 1
	RIGHT = 2
	UP = 3
	DOWN = 4
    
	def __init__(self, path = target_directory, annotations = annotation_directory, zoom_height_min = 300, zoom_height_max = 500, zoom_width_min = 500, zoom_width_max = 900, is_greyscale = 1 ):

		self.path = path

		#Coordinates for the zoom square
		self.x1 = zoom_width_min
		self.x2 = zoom_width_max

		self.y1 = zoom_height_min
		self.y2 = zoom_height_max

		self.center_x = self.x2-self.x1 / 2
		self.center_y = self.y2 -self.y1 / 2

		self.step = 0

		self.directory = os.listdir(path)
		self.annotation_list = annotation_list(annotations)
		print(len(self.annotation_list))
		self.color = 1 if is_greyscale else 3
		self.value1 = self.x2 -self.x1
		self.observation_space_values = (self.x2 -self.x1, self.y2-self.y1, self.color)
		
	def get_camera_coordinates(self):
	    return self.x1, self.x2, self.y1, self.y2
	   
	def get_observation_space_value(self):
		obs_spc_value = 1
		obs_spc = self.observation_space
		for i in obs_spc:
			obs_spc_value *= i
		return obs_spc_value
		
	def get_observation_space_values(self):
		obs_spc = self.observation_space_values
		return obs_spc

	def reset(self):
		observation = np.array([])
		self.step = 0
		done = False
		im_path = self.directory[self.step]
		im_file = self.path + "/" + im_path
		print(im_file)
		
		im = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)


		#Now we crop the image
		#If we consider (0,0) as top left corner of image called im with left-to-right
		# as x direction and top-to-bottom as y direction. and we have (x1,y1) as the top-left vertex and (x2,y2) 
		#as the bottom-right vertex of a rectangle region within that image, then:
		# roi = im[y1:y2, x1:x2]
		# X.append(im[0:100,0:100].copy()) # This will keep only the crops in the memory. 
									 # im's will be deleted by gc.
		observation = np.array(im[self.y1:self.y2, self.x1:self.x2]) # This is what the DQNAgent sees
		#cv2.imshow('image', observation)
		#k = cv2.waitKey(5000)
		#cv2.destroyAllWindows()

		
		return observation                             
                                     
	def steps_demo(self, action):
		if (action == self.LEFT):
			self.x1 -= self.PLAYER_MOVE
			self.x2 -= self.PLAYER_MOVE
		elif (action == self.RIGHT):
			self.x1 += self.PLAYER_MOVE
			self.x2 += self.PLAYER_MOVE
		elif (action == self.UP):
			self.y1 -= self.PLAYER_MOVE
			self.y2 -= self.PLAYER_MOVE
		elif (action == self.DOWN):
			self.y1 += self.PLAYER_MOVE
			self.y2 += self.PLAYER_MOVE
			
            
            
		#To avoid the zoom to get out of the image	
		if self.x1 < 0:
			self.x1 =0
			self.x2 += self.PLAYER_MOVE
			
		if self.x2 > VIDEO_RESOLUTION_X:
			self.x2 = VIDEO_RESOLUTION_X
			self.x1 -= self.PLAYER_MOVE
            
		if self.y1 < 0:
			self.y1 =0
			self.y2 += self.PLAYER_MOVE
			
		if self.y2 > VIDEO_RESOLUTION_Y:
			self.y2 = VIDEO_RESOLUTION_X
			self.y1 -= self.PLAYER_MOVE
    
            

		# We calculate the center again	
		self.center_x = self.x2-self.x1 / 2
		self.center_y = self.y2 -self.y1 / 2
		return self.x1, self.x2, self.y1, self.y2
        
        
	def steps(self, action):
		#Following the idea of openAI gym an step in our system will be moving the zoom and go to the next frame
		new_observation = np.array([])
		self.step += 1
		done = False
		if self.step > len(self.directory):
			done = True

		if (action == self.LEFT):
			self.x1 -= self.PLAYER_MOVE
			self.x2 -= self.PLAYER_MOVE
		elif (action == self.RIGHT):
			self.x1 += self.PLAYER_MOVE
			self.x2 += self.PLAYER_MOVE
		if (action == self.UP):
			self.y1 -= self.PLAYER_MOVE
			self.y2 -= self.PLAYER_MOVE
		elif (action == self.DOWN):
			self.y1 += self.PLAYER_MOVE
			self.y2 += self.PLAYER_MOVE
            
            
		#To avoid the zoom to get out of the image	
		if self.x1 < 0:
			self.x1 =0
			self.x2 += self.PLAYER_MOVE
			
		if self.x2 > VIDEO_RESOLUTION_X:
			self.x2 = VIDEO_RESOLUTION_X
			self.x1 -= self.PLAYER_MOVE
            
		if self.y1 < 0:
			self.y1 =0
			self.y2 += self.PLAYER_MOVE
			
		if self.y2 > VIDEO_RESOLUTION_Y:
			self.y2 = VIDEO_RESOLUTION_Y
			self.y1 -= self.PLAYER_MOVE
            
		# We calculate the center again	
		self.center_x = self.x2-self.x1 / 2
		self.center_y = self.y2 -self.y1 / 2
            
		#Ground value of the ball position
		ball_coordinates = self.annotation_list[self.step-1]

		ball_x = ball_coordinates['x']
		ball_y = ball_coordinates['y']
		
		if ball_x == -1 and ball_y == -1:
			if action == 1:
				reward = 0
			else:
				reward = -10
		else:
			if FindPoint(self.x1, self.x2, self.y1, self.y2, ball_x, ball_y):
				print("ball insdie area")
				reward = 1000
			else:
				reward = abs(ball_x - self.center_x)*2 + abs(ball_y - self.center_y)*2
				reward = -reward            

		if not done:
			im_path = self.directory[self.step]
			im_file = self.path + "/" + im_path
			if step % 50 == 0:
						print(im_file)                
			im = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)

        

		#Now we crop the image
			new_observation = np.array(im[self.y1:self.y2, self.x1:self.x2]) # This is what the DQNAgent sees       
		return new_observation , reward, done
        
        #return new_observation, reward, done
			

# Agent class
class DQNAgent:
    def __init__(self, observation_space_values = OBSERVATION_SPACE_VALUES):
		# Main model
        self.observation_space_values = observation_space_values
        self.reshape = (1,) + self.observation_space_values
        self.model = self.create_model()
		# Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

		# An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

		#  tensorboard object
        log_dir = "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
		# Used to count when to update target network with main network's weights
        self.target_update_counter = 0


	

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(input_shape= self.observation_space_values,filters=64,kernel_size=(3,3),padding="same", activation="relu"))  # OBSERVATION_SPACE_VALUES = (200, 400, 1) a 200x400 greyscale image.
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(ACTION_SPACE_SIZE, activation='linear')) 
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):
		
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        samples = random.sample(self.replay_memory, MINIBATCH_SIZE)
        for sample in samples:
			# Get current states from minibatch, then query NN model for Q values
            current_state, action, reward, new_state, done = sample

            test_reshape = current_state.reshape(self.reshape)


            current_qs_list = self.target_model.predict(test_reshape)


        # Get current states from minibatch, then query NN model for Q values


        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
       


        test_reshape2 = new_state.reshape(self.reshape)


        future_qs_list = self.target_model.predict(test_reshape2)




        # Now we need to enumerate our samples
        for index, (current_state, action, reward, new_current_state, done) in enumerate(samples):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but  we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[0])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[0]           
            current_qs[action] = new_q

            # And append to our training data
            
            
            test_reshape = current_state.reshape(self.reshape)


            X= test_reshape
            y= np.array(current_qs)
            y= np.uint8(y)
            y= np.reshape(y, (1,5))
            


        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(X, y, batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard])

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter % UPDATE_TARGET_EVERY == 0:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        
    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(state)
        
    def get_reshape(self):
        return self.reshape    

if __name__ == '__main__':

    step = 1
    env = Enviroment(target_directory,annotation_directory, 0,  200,  0, 400)
    values = env.get_observation_space_values()
    agent = DQNAgent(values)
    # For stats
    ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.random.set_random_seed(1)
    MODEL_NAME = "DQN_Agent"
    # Iterate over episodes

    # Update tensorboard step every episode
    #agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    # Reset environment and get initial state
    current_state = env.reset()

    reward = 0

    # Reset flag and start iterating until episode ends
    done = False
    episodes = 0
    for episode in tqdm(range(1, 20001), ascii=True, unit='episodes'):

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            test_reshape = current_state.reshape(agent.get_reshape())

            action = np.argmax(agent.get_qs(test_reshape))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.steps(action)
        print ("Step: {}, action: {}, reward: {}".format(step, action, reward))
        print("-------------------------------")
        if step % 500 ==0:
            print(env.get_camera_coordinates())

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        processed_current_state = new_state
        step += 1
        ep_rewards.append(episode_reward)
        # Append episode reward to a list and log stats (every given number of episodes)
        if not step % AGGREGATE_STATS_EVERY:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            print("tensorboard update")
            #agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if step >= 1000:
                agent.model.save(f'models\\{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
                #agent.model.save("models\\{}__{}max_{}avg_{}min__.model".format(MODEL_NAME, max_reward, average_reward, min_reward, int(time.time())))
                print("saving the model")
        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)