import gym
import pong2
import numpy as np


MaxY = 600

ENV_OBSERVATION_SPACE_LOW= 0
ENV_OBSERVATION_SPACE_HIGH = MaxY

#acciones posibles (arriba,quieto,abajo)
ENV_ACTION_SPACE_N =3
M = 40
N = 3


 #dividimos la matriz Q en trozos para el algoritmo Q learning la idea es que la matriz va a contener un tramo del recorrido total la raqueta

DISCRETE_RACKET_SIZE = M
discrete_racket_win_size = (MaxY)/DISCRETE_RACKET_SIZE




LEARNING_RATE = 0.1
#how important we measure future action over current action
DISCOUNT = 0.95
#episodios que vamos a correr el agente
EPISODES = 25000


def get_discrete_state(state):
	discrete_state = (state - ENV_OBSERVATION_SPACE_LOW) / discrete_racket_win_size
	return int(discrete_state)

q_table= np.random.uniform(low = -2, high = 0, size = (M,ENV_ACTION_SPACE_N))


