import random
import csv
import os
from pathlib import Path
from tabulate import tabulate
from abc import abstractmethod
import keras.layers as Kl
import keras.models as Km
import numpy as np
import matplotlib.pyplot as plt


class TicTacToe():

    def __init__(self, player1, player2, exp1=1, exp2=1,token='X'):
        self.state = '123456789'

        player1 = globals()[player1]
        player2 = globals()[player2]
        if token == 'X':
            self.player1 = player1(tag='X', exploration_factor=exp1)
            self.player2 = player2(tag='O', exploration_factor=exp2)
        if token == 'O':
            self.player1 = player1(tag='O', exploration_factor=exp1)
            self.player2 = player2(tag='X', exploration_factor=exp2)

        self.winner = None

    def play_game(self, optionState):            
        new_state = self.player2.make_move(optionState, self.winner)
        return new_state 
   


class Player():

    def __init__(self, tag, exploration_factor=1):
        self.tag = tag
        self.print_value = False
        self.exp_factor = exploration_factor

    def make_move(self, state, winner):
        idx = int(input('Choose move number: '))
        s = state[:idx-1] + self.tag + state[idx:]
        return s


class Agent(Player):

    def __init__(self, tag, exploration_factor=1):
        super().__init__(tag, exploration_factor)
        self.epsilon = 0.1
        self.alpha = 0.5
        self.prev_state = '123456789'
        self.state = None
        self.print_value = False

        if self.tag == 'X':
            self.op_tag = 'O'
        else:
            self.op_tag = 'X'

    @abstractmethod
    def calc_value(self, state):
        pass


    def make_move(self, state, winner):

        self.state = state

        if winner is not None:
            new_state = state
            return new_state

        p = random.uniform(0, 1)
        if p < self.exp_factor:
            new_state = self.make_optimal_move(state)
        else:
            moves = [s for s, v in enumerate(state) if v.isnumeric()]
            idx = random.choice(moves)
            new_state = state[:idx] + self.tag + state[idx + 1:]

        return new_state
    
    def make_optimal_move(self, state):
        moves = [s for s, v in enumerate(state) if v.isnumeric()]

        if len(moves) == 1:
            temp_state = state[:moves[0]] + self.tag + state[moves[0] + 1:]
            new_state = temp_state
            return new_state

        temp_state_list = []
        v = -float('Inf')

        for idx in moves:

            v_temp = []
            temp_state = state[:idx] + self.tag + state[idx + 1:]

            moves_op = [s for s, v in enumerate(temp_state) if v.isnumeric()]
            for idy in moves_op:
                temp_state_op = temp_state[:idy] + self.op_tag + temp_state[idy + 1:]
                v_temp.append(self.calc_value(temp_state_op))

            # delets Nones
            v_temp = list(filter(None.__ne__, v_temp))

            if len(v_temp) != 0:
                v_temp = np.min(v_temp)
            else:
                # encourage exploration
                v_temp = 1

            if v_temp > v:
                temp_state_list = [temp_state]
                v = v_temp
            elif v_temp == v:
                temp_state_list.append(temp_state)

        try:
            new_state = random.choice(temp_state_list)
        except ValueError:
            print('temp state:', temp_state_list)
            raise Exception('temp state empty')

        return new_state



class DeepAgent(Agent):
    def __init__(self, tag, exploration_factor=1):
        super().__init__(tag, exploration_factor)
        self.tag = tag
        self.value_model = self.load_model()

    @staticmethod
    def state2array(state):

        num_state = []
        for s in state:
            if s == 'X':
                num_state.append(1)
            elif s == 'O':
                num_state.append(-1)
            else:
                num_state.append(0)
        num_state = np.array([num_state])
        return num_state
    
    def load_model(self):
        s = 'model_values' + self.tag + '.h5'
        model = Km.load_model(s)
        return model
    
    def calc_value(self, state):
        return self.value_model.predict(self.state2array(state))

   