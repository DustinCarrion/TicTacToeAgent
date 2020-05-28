import cv2
import numpy as np
from random import randint


def cut(img):
    aux = 5
    top = 0
    total = len(img[0])
    for i in range(len(img)//5):
        if np.sum(img[i]==255) >= total-aux:
            top = i+1
            
    bottom = len(img)-1
    for i in range(len(img)-1, len(img)-(len(img)//5), -1):
        if np.sum(img[i]==255) >= total-aux:
            bottom = i
    
    left = 0
    total = len(img)
    for i in range(len(img[0])//5):
        if np.sum(img[:,i]==255) >= total-aux:
            left = i+1
            
    right = len(img[0])-1
    for i in range(len(img[0])-1, len(img[0])-(len(img[0])//5), -1):
        if np.sum(img[:,i]==255) >= total-aux:
            right = i
            
    return img[top:bottom,left:right]


def createBoard(board):
    image = np.ones((300, 300)) * 255
    cv2.line(image, (100, 0), (100, 300), (0, 255, 0), thickness=5)
    cv2.line(image, (200, 0), (200, 300), (0, 255, 0), thickness=5)
    cv2.line(image, (0, 100), (300, 100), (0, 255, 0), thickness=5)
    cv2.line(image, (0, 200), (300, 200), (0, 255, 0), thickness=5)
    token_positions = [[(5,90), (110,90), (210,90)], [(5,192), (110,192), (210,192)], [(5,294), (110,294), (210,294)]]
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] != " ":        
                cv2.putText(image,board[i][j],token_positions[i][j], cv2.FONT_HERSHEY_SIMPLEX, 4,(0,255,0),5,cv2.LINE_AA)
    return image


def isEmpty(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] != " ":
                return False
    return True


def valid(board, player):
    x = 0
    o = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] == "X":
                x+=1
            elif board[i][j] == "O":
                o+=1
    if x == o and x == 0:
        return True
    if player == "O":
        if x == o:
            return True
    else:
        if x == (o+1):
            return True
    return False

def getPossibleActions(board):
    possible_actions = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                possible_actions .append((i,j))
    return possible_actions 


def getAction(board):
    possible_actions = getPossibleActions(board)
    action = possible_actions[randint(0,len(possible_actions)-1)]
    return action 


def checkWin(board, mark):
    for i in range(3):
        if (board[i][0]==mark and board[i][1]==mark and board[i][2]==mark):
            return True
        if (board[0][i]==mark and board[1][i]==mark and board[2][i]==mark):
            return True
    if (board[0][0] == mark and board[1][1]==mark and board[2][2]==mark):
        return True
    if (board[2][0] == mark and board[1][1]==mark and board[0][2]==mark):
        return True
    return False


def checkTie(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                return False
    else:
        return True


def play_q_agent(board, action, player):
    board[action[0]][action[1]] = player
    if checkWin(board, player):
        return board, True, False
    if checkTie(board):
        return board, False, True
    return board, False, False

def play_neural_agent(agent, board, player):
    board = agent.play_game(optionState=board)
    board = transformAgentBoard(board)
    if checkWin(board, player):
        return board, True, False
    if checkTie(board):
        return board, False, True
    return board, False, False


def transformDigitalBoard(digital_board):
    agent_board = [["1","2","3"],["4","5","6"],["7","8","9"]]
    for i in range(3):
        row = []
        for j in range(3):
            if digital_board[i][j] != " ":
                agent_board[i][j] = digital_board[i][j]
    agent_board = "".join(np.array(agent_board).flatten())
    return agent_board

def transformAgentBoard(agent_board):
    aux = "123456789"
    digital_board = ""
    for i in agent_board:
        if i in aux:
            digital_board += " "
        else:
            digital_board += i
    digital_board = np.array(list(digital_board)).reshape((3,3))
    return digital_board
