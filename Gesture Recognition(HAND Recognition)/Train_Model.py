
# coding: utf-8

# In[86]:


import os
from PIL import Image
import cv2
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,LSTM,TimeDistributed,Conv2D,MaxPooling2D,Flatten,Dropout
from sklearn.model_selection import train_test_split


# In[3]:


path="/home/gopal/Desktop/all_in one/text_genration/project_info/gesture/"


# In[4]:


class_=os.listdir(path)


# In[5]:


x_=[]
y_=[]
for classes in class_:
    cc=path+classes+"/"
    for ix in os.listdir(cc):
        img=Image.open(cc+ix)
        img=np.array(img)/255.0
        x_.append(img)
        y_.append(classes)
        
    
  
     
X=np.array(x_)
Y=np.array(y_)


# In[39]:


X=np.reshape(X,(4800,1,50,50,1)) # TimeDistributed in cnn take 5D


# In[40]:


X.shape


# In[41]:


from keras.utils import np_utils


# In[42]:


y=np_utils.to_categorical(Y)


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[50]:


model=Sequential()
model.add(TimeDistributed(Conv2D(64,(3,3),activation="relu"),input_shape=(None,50,50,1)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(4,4))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(10))
model.add(Dense(4,activation="softmax"))
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])
model.summary()


# In[90]:


model.fit(X_train,y_train,epochs=5,batch_size=32,verbose=2,shuffle=True,validation_data=[X_test,y_test])


# In[85]:


model.save("mygesture.h5")

