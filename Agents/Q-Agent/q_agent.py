from ticTacToeEnvironment import TicTacToe
from random import uniform
from copy import deepcopy
import numpy as np
import joblib

#Defining the parameters
alpha = 0.1
gamma = 0.6
threshold = 0.1
env = TicTacToe()


#Training player 1 
states_player1 = []
q_table_player1 = []
for i in range(1,1000001):
    env.initGame()
    while True:
        if env.board not in states_player1:
            states_player1.append(deepcopy(env.board))
            q_table_player1.append([0,0,0,0,0,0,0,0,0])
            action = env.getAction() 
            max_prob = 0
            index_state = -1
            index_max_prob = env.actions.index(action)
        else:   
            index_state = states_player1.index(env.board)
            if uniform(0, 1) < threshold:
                action = env.getAction()
                index_max_prob = env.actions.index(action)
                max_prob = q_table_player1[index_state][index_max_prob]
            else:
                possible_actions = env.getPossibleActions()
                probabilities_possible_actions = []
                indexes_possible_actions = []
                for action in possible_actions:
                    action_index = env.actions.index(action)
                    probabilities_possible_actions.append(q_table_player1[index_state][action_index])
                    indexes_possible_actions.append(action_index)
                max_prob = max(probabilities_possible_actions)
                index_max_prob = indexes_possible_actions[probabilities_possible_actions.index(max_prob)]
                action = env.actions[index_max_prob]

        reward_player1, end_game = env.play(action, 1, 1)  
                
        if end_game:   
            old_value = max_prob
            next_max = 0
            new_value = (1 - alpha) * old_value + alpha * (reward_player1 + gamma * next_max)
            q_table_player1[index_state][index_max_prob] = new_value
            break
        
        reward_player2, end_game = env.playRandom(2, 1)
        
        if env.board not in states_player1:
            states_player1.append(deepcopy(env.board))
            q_table_player1.append([0,0,0,0,0,0,0,0,0])
                
        total_reward = reward_player1 + reward_player2
        old_value = max_prob
        index_next_state = states_player1.index(env.board)
        next_max = max(q_table_player1[index_next_state])
        new_value = (1 - alpha) * old_value + alpha * (total_reward + gamma * next_max)
        q_table_player1[index_state][index_max_prob] = new_value

        if end_game:
            break

    if i % 100000 == 0:
        print(f"Episode: {i}")
print("Training finished.\n")
joblib.dump(q_table_player1, 'q_table_player1.sav')
joblib.dump(states_player1, 'states_player1.sav')


#Training player 2
states_player2 = []
q_table_player2 = []
for i in range(1,1000001):
    env.initGame()
    reward_player1, end_game = env.playRandom(1, 2)
    if env.board not in states_player2:
        states_player2.append(deepcopy(env.board))
        q_table_player2.append([0,0,0,0,0,0,0,0,0])
    while True:
        index_state = states_player2.index(env.board)
        if uniform(0, 1) < threshold:
            action = env.getAction()
            index_max_prob = env.actions.index(action)
            max_prob = q_table_player2[index_state][index_max_prob]
        else:
            possible_actions = env.getPossibleActions()
            probabilities_possible_actions = []
            indexes_possible_actions = []
            for action in possible_actions:
                action_index = env.actions.index(action)
                probabilities_possible_actions.append(q_table_player2[index_state][action_index])
                indexes_possible_actions.append(action_index)
            max_prob = max(probabilities_possible_actions)
            index_max_prob = indexes_possible_actions[probabilities_possible_actions.index(max_prob)]
            action = env.actions[index_max_prob]
           
        reward_player2, end_game = env.play(action, 2, 2)
        if end_game:
            old_value = max_prob
            next_max = 0
            new_value = (1 - alpha) * old_value + alpha * (reward_player2 + gamma * next_max)
            q_table_player2[index_state][index_max_prob] = new_value
            break
        
        reward_player1, end_game = env.playRandom(1, 2)
        
        if env.board not in states_player2:  
            states_player2.append(deepcopy(env.board))
            q_table_player2.append([0,0,0,0,0,0,0,0,0])
        
        total_reward = reward_player1 + reward_player2
        old_value = max_prob
        index_next_state = states_player2.index(env.board)
        next_max = max(q_table_player2[index_next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (total_reward + gamma * next_max)
        q_table_player2[index_state][index_max_prob] = new_value

        if end_game:
            break
        
    if i % 100000 == 0:
        print(f"Episode: {i}")
print("Training finished.\n")
joblib.dump(q_table_player2, 'q_table_player2.sav')
joblib.dump(states_player2, 'states_player2.sav')   





