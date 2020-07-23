import gym
import numpy as np

env = gym.make('MountainCar-v0')

while not done:
	env.reset()
	cnt += 1
	action = env.action_space.sample() # random action
	observation, reward, done, info = env.step(action)
	if done:
	    break

print("game lasted ", cnt, "moves.")

