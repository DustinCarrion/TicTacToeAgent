from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

import numpy as np
import cv2
from os import listdir

sc = StandardScaler()

X_train = []
y_train = []
training_paths = ["train/no_board", "train/board"]
label = 0
for path in training_paths:
    train_images = listdir(path)
    for image_name in train_images:
        image = cv2.imread(f"{path}/{image_name}", 0)
        image = cv2.resize(image, (50,50))
        X_train.append(image.flatten())
        y_train.append(label)
    label += 1
X_train = sc.fit_transform(X_train)        

X_test = []
y_test = []
testing_paths = ["test/no_board", "test/board"]
label = 0
for path in testing_paths:
    train_images = listdir(path)
    for image_name in train_images:
        image = cv2.imread(f"{path}/{image_name}", 0)
        image = cv2.resize(image, (50,50))
        X_test.append(image.flatten())
        y_test.append(label)
    label += 1
X_test = sc.transform(X_test)
    
clf = SVC(C=10, kernel="rbf", tol=0.001, gamma="scale", probability=True)
clf.fit(X_train, y_train)
print(confusion_matrix(y_test, clf.predict(X_test)))