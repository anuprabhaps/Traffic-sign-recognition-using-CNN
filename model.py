import numpy as np
import pandas as pd
import matplotlib.pyplot as plt#to plot accuracy
import cv2
import tensorflow
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
path="C:/Users/anuraj/Downloads/archive/myData"
features=[]
target=[]
print('Importing Folders...')
for i in range(43):
    folders=os.listdir(path+'/'+str(i))
    for name in folders:
        Img=cv2.imread(path+'/'+str(i)+'/'+name) #image as array
        features.append(Img)
        target.append(i)
    print(i,end=" ")

data = np.array(features)
labels = np.array(target)
print(data.shape, labels.shape)
#Splitting training and testing dataset
X_t1, X_t2, y_t1, y_t2 = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_t1.shape, X_t2.shape, y_t1.shape, y_t2.shape)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(32,32,3)))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_t1, y_t1, batch_size=32, epochs=10, validation_data=(X_t2, y_t2))

plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.show()
y_pred=model.predict(X_t2)

pred_class=np.argmax(y_pred,axis=1)
pred_class
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_t2,pred_class)
import seaborn as sns

labels=pd.read_csv("C:/Users/anuraj/Downloads/archive/labels.csv")

plt.subplots(figsize=(35,30)) 
sns.heatmap(cm,annot=True,fmt="d",xticklabels=labels["Name"],yticklabels=labels["Name"],cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title("Confusion Matrix", fontsize=20)
model.save("model.h5")
