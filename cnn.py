#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator


# In[2]:


train_datagen=ImageDataGenerator(rescale=1/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1/255)


# In[3]:


x_train=train_datagen.flow_from_directory(r'/users/apurvaaddula/Desktop/chest_xray/train',target_size=(64,64),batch_size=32,class_mode='binary')
x_test=test_datagen.flow_from_directory(r"/users/apurvaaddula/Desktop/chest_xray/test",target_size=(64,64),batch_size=32,class_mode='binary')


# In[4]:


print(x_train.class_indices)


# In[5]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


# In[6]:


model=Sequential()


# In[7]:


model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))


# In[8]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[9]:


model.add(Flatten())


# In[10]:


model.add(Dense(output_dim=128,init='random_uniform',activation='relu'))


# In[11]:


model.add(Dense(output_dim=1,init='random_uniform',activation='sigmoid'))


# In[12]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[13]:


import tensorflow as tf
tf.compat.v1.global_variables


# In[14]:


model.fit_generator(x_train,steps_per_epoch=163,epochs=10,validation_data=x_test,validation_steps=20)


# In[15]:


model.save('cnn.h5')


# In[16]:


from keras.models  import load_model
from keras.preprocessing import image
import numpy as np


# In[17]:


model=load_model("cnn.h5")


# In[18]:


img=image.load_img(r"/users/apurvaaddula/Desktop/IM-0001-0001 copy.jpeg",target_size=(64,64))


# In[19]:


x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)


# In[20]:


pred=model.predict_classes(x)


# In[21]:


pred


# In[22]:


index=['Normal','Pneumonia']


# In[ ]:




