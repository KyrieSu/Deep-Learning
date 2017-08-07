import numpy as np
from keras.datasets import cifar10

np.random.seed(10)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

from keras.utils import np_utils

y_train_1hot = np_utils.to_categorical(y_train)
y_test_1hot = np_utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dropout,Dense,Flatten,Conv2D,MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history = model.fit(x_train,y_train_1hot,batch_size=128,epochs=10,verbose=1,validation_split=0.2)

from show import show_train_history

show_train_history('acc','val_acc')
show_train_history('loss','val_loss')

scores = model.evaluate(x_test,y_test_1hot,verbose=0)
print (scores)

pred = model.predict_classes(x_test)
print(pred[:10])

from show import plot_images_labels_prediction
plot_images_labels_prediction(x_test,y_test,pred,0,10)