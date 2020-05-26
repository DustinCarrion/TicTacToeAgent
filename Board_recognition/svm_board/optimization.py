# -*- coding: utf-8 -*-
import gc

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def read_file():
    return joblib.load("X_train.sav"), joblib.load("X_test.sav"), joblib.load("y_train.sav"), joblib.load("y_test.sav") 

def calculateAccuracy(matrix):
    total = 0
    failures = 0
    for i in range(len(matrix)):
        total += sum(matrix[i])
        for j in range(len(matrix[0])):
            if i != j:
                failures += matrix[i,j]
    accuracy = ((total-failures)/total)*100
    return accuracy


def checkParameters(parametersToSave, savedParameters, ab=False):
    if ab:
        for i in savedParameters:
            if i[1:] == parametersToSave[1:] and type(i[0]) == type(parametersToSave[0]):
                return True
        return False
    else:
        if parametersToSave in savedParameters:
            return True
        return False
            

if __name__ == '__main__':
    np.random.seed(3)

    X_train, X_test, y_train, y_test = read_file()
    #--------------------SVM------------------
    C = [0.5, 1, 10, 50, 100, 200, 300] 
    kernel = ['linear', 'rbf', 'sigmoid']
    tol = [1e-3, 0.1, 1, 1e-5, 1e-6, 1e-7] 
    gamma = ['scale', 'auto'] 

    acc = [] 
    parameters = []
    for i in range(len(C)):
        for j in range(len(kernel)):
            for k in range(len(tol)):
                for l in range(len(gamma)):
                    if checkParameters([C[i],kernel[j],tol[k],gamma[l]], parameters):
                        continue
                    clf = SVC(C=C[i], kernel=kernel[j], tol=tol[k], gamma=gamma[l], random_state=2)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test) 
                    acc.append(calculateAccuracy(confusion_matrix(y_test,y_pred)))
                    parameters.append([C[i],kernel[j],tol[k],gamma[l]])
                    print(C[i],kernel[j],tol[k],gamma[l])
                    gc.collect()
                    # joblib.dump(acc, f"results/lvl_{decomposition_level}/acc_svm.sav")
                    # joblib.dump(parameters, f"results/lvl_{decomposition_level}/parameters_svm.sav")
