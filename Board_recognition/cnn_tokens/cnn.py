from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import numpy as np
import cv2
from os import listdir
from sklearn.preprocessing import OneHotEncoder

X_train = None
y_train = []
training_paths = ["_", "o", "x"]
label = 0
for path in training_paths:
    train_images = listdir(path)
    for image_name in train_images:
        image = cv2.imread(f"{path}/{image_name}", 0)
        image = cv2.resize(image, (200,200))
        image = image.reshape(1,200,200,1)
        if not isinstance(X_train, np.ndarray):
            X_train = image
        else:
            X_train = np.concatenate((X_train,image),axis=0)
        y_train.append(label)
    label += 1
enc = OneHotEncoder(categories="auto")
y_train = np.array(y_train).reshape(len(y_train), 1)
y_train = enc.fit_transform(y_train).toarray()
        

model = Sequential() 

model.add(Conv2D(20, (3, 3), input_shape = (200, 200, 1), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(10, (3, 3), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (3, 3)))

model.add(Flatten()) 
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 32, activation = 'relu')) 
model.add(Dense(units = 3, activation = 'softmax')) 

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) 
model.fit(X_train, y_train, epochs=50)

model.save('cnn_tokens.h5') 
