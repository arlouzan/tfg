import gym
import numpy as np
from gym import wrappers

env = gym.make('MountainCar-v0')

bestlength=0
episode_lenghts = []

best_weights = np.zeros(4)

for i in range(100):
	new_weights= np.random.uniform(-1.0, 1.0, 4)
	length = []
	
	for j in range(100):
		observation = env.reset()
		done = False
		cnt = 0
		while not done:
			#env.render()
			cnt += 1
			action = 1 if np.dot(observation, new_weights) > 0 else 0
			observation, reward, done, info = env.step(action)
			if done:
			    break
		lenght.append(cnt)
	average_length = float(sum(length)/ len(length))
	if average_length > bestlength:
		bestlength = average_length
		best_weights = new_weights
	episode_lengths.append(average_lenght)
	if i % 10 == 0:
		print("best lenght is  ", bestlength) 
done = False
cnt = 0
env = wrappers.Monitor(env, "TestMovies", force=True)
observation = env.reset()
while not done:
	#env.render()
	cnt += 1
	action = 1 if np.dot(observation, new_weights) > 0 else 0
	observation, reward, done, info = env.step(action)
	if done:
	    break

print("game lasted ", cnt, "moves.")


