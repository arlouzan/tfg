import os
import cv2
import PIL.Image
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


target_directory = "/home/shadowadri/practice_tfg/dataset/test/images/test_6"

annotation_directory = "/home/shadowadri/practice_tfg/dataset/test/annotations/test_6/ball_markup.json"



OBSERVATION_SPACE_VALUES = (200,400,1)
ACTION_SPACE_SIZE = 3 #left, stand still, right

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


def pre_process_image(image):
    """Pre-process a raw image from the game-environment."""

    # Convert image to gray-scale.
    #img_gray = _rgb_to_grayscale(image=image)
	
	#Comment if the image is in RGB and uncomment the previous line
    img_gray = image
	
    # Create PIL-object from numpy array.
    img = PIL.Image.fromarray(img_gray)
    # Resize the image.
#    img_resized = img.resize(resample=PIL.Image.LINEAR)

    # Convert 8-bit pixel values back to floating-point.
    img = np.float32(img)	

    return img


DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 200  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 100  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'test'
MIN_REWARD = -100  # For model save


# Environment settings
EPISODES = 500

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes

# Own Tensorboard class. This allows to have just one log file for all fit()'s
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class Enviroment:

	OFF_CENTER_PENALTY = 10
	PLAYER_MOVE = 5
	ACTION_SPACE_SIZE = 3 # Move left, stay still, move right
	LEFT = 0
	NO_OP = 1
	RIGHT = 3
    
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
		self.color = 1 if is_greyscale else 3

		self.observation_space_values = (self.x2 -self.x1, self.y2-self.y1, self.color)

	   
	def get_observation_space_value(self):
		obs_spc_value = 1
		obs_spc = self.observation_space
		for i in obs_spc:
			obs_spc_value *= i
		return obs_spc_value
		
	def get_observation_space_values(self):
		obs_spc_value = 1
		obs_spc = self.observation_space
		return obs_spc
			
		
	def reset(self):
		observation = np.array([])
		self.step = 0
		done = False
		im_path = self.directory[self.step]
		im_file = self.path + "/" + im_path
		print("getting the image....")
		print("Path : ", im_file)
		
		
		#This part was giving me some error with the image scale, trying different fucntions to solve the problem
		
		
		im = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
		print("*****************************************")
		print ('img.dtype: ', im.shape)
		print ('img.dtype: ', im.dtype)
		print ('img.size: ', im.size)

		#Now we crop the image
		#If we consider (0,0) as top left corner of image called im with left-to-right
		# as x direction and top-to-bottom as y direction. and we have (x1,y1) as the top-left vertex and (x2,y2) 
		#as the bottom-right vertex of a rectangle region within that image, then:
		# roi = im[y1:y2, x1:x2]
		# X.append(im[0:100,0:100].copy()) # This will keep only the crops in the memory. 
									 # im's will be deleted by gc.
		observation = np.array(im[self.y1:self.y2, self.x1:self.x2]) # This is what the DQNAgent sees
		print ('observation.dtype: ', observation.shape)
		print ('observation.dtype: ', observation.dtype)
		print ('observation.size: ', observation.size)	
		print("************RESET DONE******************")
		
		
		return observation                             
                                     
		
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
		
		# We calculate the center again	
		self.center_x = self.x2-self.x1 / 2
		self.center_y = self.y2 -self.y1 / 2
		
		#Ground value of the ball position
		ball_coordinates = self.annotation_list[self.step-1]
		
		ball_x = ball_coordinates['x']
		ball_y = ball_coordinates['y']
		
		reward = abs(ball_x - self.center_x)**2
		
		if reward <= self.OFF_CENTER_PENALTY:
			reward = 200
		else:
			reward *= -1
		
		if not done:
			im_path = self.directory[self.step]
			im_file = self.path + "/" + im_path

		im = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)


		#Now we crop the image
		#If we consider (0,0) as top left corner of image called im with left-to-right
		# as x direction and top-to-bottom as y direction. and we have (x1,y1) as the top-left vertex and (x2,y2) 
		#as the bottom-right vertex of a rectangle region within that image, then:
		# roi = im[y1:y2, x1:x2]
		# X.append(im[0:100,0:100].copy()) # This will keep only the crops in the memory. 
									 # im's will be deleted by gc.
		new_observation = np.array(im[self.y1:self.y2, self.x1:self.x2]) # This is what the DQNAgent sees
		print ('new_observation.dtype: ', new_observation.shape)
		print ('new_observation.dtype: ', new_observation.dtype)
		print ('new_observation.size: ', new_observation.size)
		print("*****************************++END STEPS++******************************")
		pre_process_image(new_observation)
		
		return new_observation, reward, done
			
			
		
	
# Agent class
class DQNAgent:
    def __init__(self):
		

		
        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        

	

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(input_shape=OBSERVATION_SPACE_VALUES,filters=64,kernel_size=(3,3),padding="same", activation="relu"))  # OBSERVATION_SPACE_VALUES = (200, 400, 1) a 200x400 greyscale image.
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
		
        print("+++++++++++++++++++ENTERING THE HYPE TRAIN. CHOOOOO CHOOOO MOTHERFUCKER+++++++++++++++++++++++++++++++++")

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        samples = random.sample(self.replay_memory, MINIBATCH_SIZE)
        for sample in samples:
			# Get current states from minibatch, then query NN model for Q values
            current_state, action, reward, new_state, done = sample
            processed_current_state=pre_process_image(current_state)
            processed_current_state= np.uint8(processed_current_state)
            processed_current_state= np.reshape(processed_current_state, (1,200,400,1))
            processed_current_state= np.uint8(processed_current_state)
            
            print ('processed_current_state.shape: ', processed_current_state.shape)
            print ('processed_current_state.dtype: ', processed_current_state.dtype)
            print ('processed_current_state.size: ', processed_current_state.size)

            current_qs_list = self.target_model.predict(processed_current_state)
            print("*****************************++END TRAIN++******************************")

        # Get current states from minibatch, then query NN model for Q values


        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        
        processed_new_state=pre_process_image(new_state)
        processed_new_state= np.uint8(processed_new_state)
        print ('processed_new_state.shape: ', processed_new_state.shape)
        print ('processed_new_state.dtype: ', processed_new_state.dtype)
        print ('processed_new_state.size: ', processed_new_state.size)
        processed_new_state= np.reshape(processed_new_state, (1,200,400,1))
        processed_new_state= np.uint8(processed_new_state)
        
        future_qs_list = self.target_model.predict(processed_new_state)


        print ('future_qs_list.shape: ', future_qs_list.shape)
        print ('future_qs_list.dtype: ', future_qs_list.dtype)
        print ('future_qs_list.size: ', future_qs_list.size)

        X = []
        y = []

        # Now we need to enumerate our samples
        for index, (current_state, action, reward, new_current_state, done) in enumerate(samples):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]


agent = DQNAgent()

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1
    env = Enviroment()
    # Reset environment and get initial state
    current_state = env.reset()
    processed_current_state = pre_process_image(current_state)

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(processed_current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.steps(action)
        processed_new_state = pre_process_image(new_state)
        

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        # Every step we update replay memory and train main network
        agent.update_replay_memory((processed_current_state, action, reward, processed_new_state, done))
        agent.train(done, step)

        processed_current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            #agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
            agent.model.save("models/{}__{}max_{}avg_{}min__.model".format(MODEL_NAME, max_reward, average_reward, min_reward, int(time.time())))
    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

