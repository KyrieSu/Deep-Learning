import numpy as np
from keras.utils import np_utils

np.random.seed(10)

from keras.datasets import mnist

(x_train , y_train) , (x_test,y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

y_train_1hot = np_utils.to_categorical(y_train)
y_test_1hot = np_utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense , Dropout
model = Sequential()
model.add(Dense(units=1000, input_dim=784,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=x_train,y=y_train_1hot,validation_split=0.2,epochs=10, batch_size=200,verbose=2)

from show import show_train_history

show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')
scores = model.evaluate(x_test, y_test_1hot)
print()
print('accuracy=',scores[1])
prediction = model.predict_classes(x_test)

import pandas as pd

pd.crosstab(y_test,prediction,rownames=['label'],colnames=['pred'])
df = pd.DataFrame({'label' : y_test , 'pred' : prediction})

true5but3 = df[(df.label==5)and(df.label==3)]
