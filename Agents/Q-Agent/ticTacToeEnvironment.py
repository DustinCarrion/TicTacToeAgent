# -*- coding: utf-8 -*-
#Importing the libraries

from random import randint
import joblib


#Creating the environment
class TicTacToe:
    def __init__(self, player1="X", player2="O"):
        self.player1 = player1
        self.player2 = player2
        self.board = None
        self.actions = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    
    def initGame(self):
        self.board = []
        for i in range(3):
            row = []
            for j in range(3):
                row.append(' ')
            self.board.append(row)
    
    def printBoard(self):
        empty_board = "      ___ ___ ___\n     |   |   |   |\n     | 0 | 1 | 2 |\n  ___|___|___|___|\n |   |   |   |   |\n | 0 |   |   |   |\n |___|___|___|___|\n |   |   |   |   |\n | 1 |   |   |   |\n |___|___|___|___|\n |   |   |   |   |\n | 2 |   |   |   |\n |___|___|___|___|\n"
        '''   ___ ___ ___
             |   |   |   |
             | 0 | 1 | 2 |
          ___|___|___|___|
         |   |   |   |   |
         | 0 | X |   | O |
         |___|___|___|___|
         |   |   |   |   |
         | 1 | d | e | f |
         |___|___|___|___|
         |   |   |   |   |
         | 2 | g | h | i |
         |___|___|___|___|'''
        a = self.board[0][0] 
        b = self.board[0][1] 
        c = self.board[0][2] 
        d = self.board[1][0] 
        e = self.board[1][1] 
        f = self.board[1][2] 
        g = self.board[2][0] 
        h = self.board[2][1] 
        i = self.board[2][2] 
        current_board = empty_board[0:101] + a + empty_board[102:105] + b + empty_board[106:109] + c + empty_board[110:158] + d + empty_board[159:162] + e + empty_board[163:166] + f + empty_board[167:215] + g + empty_board[216:219] + h + empty_board[220:223] + i + empty_board[224:]
        print(current_board)
        print("-"*30)

    def checkWin(self, player):
        mark = 'X' if player==1 else 'O'
        for i in range(3):
            if (self.board[i][0]==mark and self.board[i][1]==mark and self.board[i][2]==mark):
                return True
            if (self.board[0][i]==mark and self.board[1][i]==mark and self.board[2][i]==mark):
                return True
        if (self.board[0][0] == mark and self.board[1][1]==mark and self.board[2][2]==mark):
            return True
        if (self.board[2][0] == mark and self.board[1][1]==mark and self.board[0][2]==mark):
            return True
        return False

    def checkTie(self):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    return False
        else:
            return True

    def getPossibleActions(self):
        possible_actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    possible_actions .append((i,j))
        return possible_actions 

    def getAction(self):
        possible_actions = self.getPossibleActions()
        action = possible_actions[randint(0,len(possible_actions)-1)]
        return action

    def calculateReward(self, player, player_training):
        if self.checkWin(player):
            if player_training == 1:
                if player == 1:
                    return 20, True
                else:
                    return -10, True
            else:
                if player == 2:
                    return 20, True
                else:
                    return -10, True
        elif self.checkTie():
            return -10, True
        else:
            return -1, False
    
    def play(self, action, player, player_training):
        self.board[action[0]][action[1]] = 'X' if player == 1 else 'O'
        return self.calculateReward(player, player_training)

    def playRandom(self, player, player_training):
        action = self.getAction()
        self.board[action[0]][action[1]] = 'X' if player == 1 else 'O'
        return self.calculateReward(player, player_training)

    def verifyAction(self, action):
        possible_actions = self.getPossibleActions()
        if action in possible_actions:
            return True
        return False
        
    def playHuman(self, player):
        while True:
            action = input("Input the action: ")
            try:
                action = action.split(",")
            except:
                print("The action needs to be two numbers between 0-2. Ex: 0,1. Try again")
            try:    
                action[0] = int(action[0])
                action[1] = int(action[1])
                action = tuple(action)
                if self.verifyAction(action):
                    break
                else:
                    print("The action needs to be two numbers between 0-2. Ex: 0,1. Try again")
            except:
                print("The action needs to be two numbers between 0-2. Ex: 0,1. Try again")
        return self.play(action, player, 1)
    
    def initRealGame(self):
        self.player1 = "Q-Learning Agent"
        self.player2 = input("Input the name of the player: ")
        q_table_player1 = joblib.load("q_table_player1.sav")
        states_player1 = joblib.load("states_player1.sav")
        self.initGame()
        self.printBoard()
        _ = input(f"Press enter to start the game and good luck {self.player2} ;)")
        while True:
            possible_actions = self.getPossibleActions()
            index_state = states_player1.index(self.board)
            probabilities_possible_actions = []
            indexes_possible_actions = []
            for action in possible_actions:
                action_index = self.actions.index(action)
                probabilities_possible_actions.append(q_table_player1[index_state][action_index])
                indexes_possible_actions.append(action_index)
            max_prob = max(probabilities_possible_actions)
            index_max_prob = indexes_possible_actions[probabilities_possible_actions.index(max_prob)]
            action = self.actions[index_max_prob]
        
            reward_player1, end_game = self.play(action, 1, 1)  
            self.printBoard()
            if end_game:
                if reward_player1 == 20:
                    print(f"{self.player1} wins!")
                if reward_player1 == -10:
                    print("Tie")
                break
            
            _ , end_game = self.playHuman(2)
            self.printBoard()
            if end_game:
                print(f"{self.player2} wins!")
                break
            
            


