import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import joblib
from DeepTicTacToe import TicTacToe
from helper_functions import *

BOARD_PREDICTION_RELIABILITY = 1.80
PADDING = 50
TOKENS = [" ", "O", "X"]
ACTION_INDEXES = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]

#--------------BOARD RECOGNIZER--------------
svm = joblib.load(f'svm.sav')
sc = joblib.load(f'sc.sav')
cnn_board = load_model(f'cnn_board.h5')
cnn_tokens = load_model(f'cnn_tokens.h5')

#--------------AGENT--------------
while True:
    player = input("Select your token to play (X, O): ")
    if player == "X" or player == "O":
        break
    print("The available tokens are X and O, plese try again.")

print("Select the type of agent that you want to play with:\n1) Neural agent\n2) Q-Agent")
while True:
    agent_type = input("Select (1, 2): ")
    if agent_type == "1" or agent_type == "2":
        break
    print("The available agents are 1 and 2, plese try again.")
if player == "O":
    agent_token = "X"
    if agent_type == "1":
        agent = TicTacToe('DeepAgent', "Player", 1, 1,token='O')
    else:
        q_table = joblib.load("q_table_player1.sav")
        states = joblib.load("states_player1.sav")
else:
    agent_token = "O"
    if agent_type == "1":
        agent = TicTacToe('Player', 'DeepAgent', 1, 1,token='X')
    else:
        q_table = joblib.load("q_table_player2.sav")
        states = joblib.load("states_player2.sav")
    


win = False
tie = False
cam = cv2.VideoCapture(0)
while not(win) and not(tie):
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    img = frame[60:420,:]
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 32)
    cv2.imshow("Input", frame)
    cv2.imshow("Filtered Input", img)
    
    k = cv2.waitKey(1)
    if k%256 == 27: #escape
        break
    elif k%256 == 32: #space
        img = cv2.copyMakeBorder(img, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT, value=255)
        roi = 250
        x=0
        y=0
        find = False
        
        while roi <= len(img):
            if find: break
            threshold = roi*roi*0.08
            
            while y+roi <= len(img):
                if find: break
                
                while x+roi <= len(img[0]):
                    if find: break
                    roi_image = img[y:y+roi, x:x+roi]
                    
                    if np.sum(roi_image == 0) > threshold:
                        svm_img = cv2.resize(roi_image, (50,50))
                        svm_img = sc.transform([svm_img.flatten()])
                        svm_prediction = svm.predict_proba(svm_img)[0]
                        
                        cnn_img = cv2.resize(roi_image, (300,300))
                        cnn_img = image.img_to_array(cnn_img)
                        cnn_img = np.expand_dims(cnn_img, axis = 0)
                        cnn_prediction = cnn_board.predict(cnn_img)[0]
                        if np.argmax(svm_prediction) == 1 and np.argmax(cnn_prediction) == 3:
                            if (svm_prediction[1] + cnn_prediction[3]) > BOARD_PREDICTION_RELIABILITY:
                                #------------------VERIFY THE BOARD------------------
                                board_found = cut(roi_image)
                                error = False
                                while True:
                                    cv2.imshow("Board Found", board_found)
                                    c = cv2.waitKey(1)
                                    if c%256 == 48: #0
                                        cv2.destroyWindow("Board Found")
                                        error = True
                                        break
                                    elif c%256 == 49: #1
                                        cv2.destroyWindow("Board Found")
                                        break
                                if error: break
                                
                                #------------------DIGITALIZE THE BOARD------------------
                                board_found = cv2.resize(board_found, (600,600))
                                digital_board = []
                                for i in range(3):
                                    row = []
                                    for j in range(3):
                                        token = board_found[i*200:(i+1)*200,j*200:(j+1)*200]
                                        token_img = image.img_to_array(token)
                                        token_img = np.expand_dims(token_img, axis = 0)
                                        cnn_token_prediction = np.argmax(cnn_tokens.predict(token_img)[0])
                                        row.append(TOKENS[cnn_token_prediction])
                                    digital_board.append(row)
                                
                                #------------------BOARD VERIFICATIONS------------------
                                if isEmpty(digital_board) and player == "X":
                                    find = True
                                    print("The board is empty. Player X must starts.")
                                    break
                                
                                win = checkWin(digital_board, player)
                                if win:
                                    find = True
                                    print("Player wins")
                                    break
                                tie = checkTie(digital_board)
                                if tie:
                                    find = True
                                    print("Tie")
                                    break
                                
                                img_board = createBoard(digital_board) 
                                while True:
                                    cv2.imshow("Digitized Board", img_board)
                                    d = cv2.waitKey(1)
                                    if d%256 == 48: #0
                                        cv2.destroyWindow("Digitized Board")
                                        break
                                    elif d%256 == 49: #1
                                        find = True
                                        cv2.destroyWindow("Digitized Board")
                                        
                                        #------------------AGENT PLAY------------------
                                        if agent_type == "1":
                                            agent_board = transformDigitalBoard(digital_board)
                                            digital_board, win, tie = play_neural_agent(agent, agent_board, agent_token)
                                        else:
                                            possible_actions = getPossibleActions(digital_board) 
                                            if digital_board in states:
                                                index_state = states.index(digital_board)
                                                probabilities_possible_actions = []
                                                indexes_possible_actions = []
                                                for action in possible_actions:
                                                    action_index = ACTION_INDEXES.index(action)
                                                    probabilities_possible_actions.append(q_table[index_state][action_index])
                                                    indexes_possible_actions.append(action_index)
                                                max_prob = max(probabilities_possible_actions)
                                                index_max_prob = indexes_possible_actions[probabilities_possible_actions.index(max_prob)]
                                                action = ACTION_INDEXES[index_max_prob]
                                            else:
                                                action = getAction(digital_board)
                                            digital_board, win, tie = play_q_agent(digital_board, action, agent_token)
                                        img_board = createBoard(digital_board) 
                                        if win: 
                                            print("Agent wins")
                                        elif tie:
                                            print("Tie")
                                        while True:
                                            cv2.imshow("Agent Move", img_board)
                                            e = cv2.waitKey(1)
                                            if e%256 == 49:
                                                cv2.destroyWindow("Agent Move")
                                                break
                                        break
                                        
                                break
                    x+=25
                x=0
                y+=25
            x=0
            y=0
            roi+=25
        if not find:
            print("Error while capturing board")
        

cam.release()

cv2.destroyAllWindows()


        
        
 

