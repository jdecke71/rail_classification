import tensorflow as tf
import  os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers.core import Dense,Droupout,Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

wabtec = 'C:/Users/jvargh81/Desktop/Wabtec/denver_1_updated.csv'
cat = 'C:/Users/jvargh81/Desktop/Wabtec/catenary/'
ncat = 'C:/Users/jvargh81/Desktop/Wabtec/not_catenary'

cat_ncat = pd.read_csv(wabtec)

cat_ncat.head()
cat_ncat['Name'].unique()

def collect_all_images(csv,training,directory,image_height,image_width) :
        for folder in directory:
            for file in csv['Name']:
                path = os.path.join(folder,file+'.png')
                img_array = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                new_image = cv2.resize(img_array,image_height,image_width)
                if csv['Catenary'] is 'yes' :
                    training.append([new_image,1])  
                else :
                    training.append([new_image,0])  
        
        return training
		
		
def generate_trainig_and_testData(training,features,labels):
    
    random.shuffle(training_data)
    
    for feature, label in training_data:
        features.append(feature)
        labels.append(label)
    
    return features, labels
	
	
def generate_tensors(features,tensor_input,image_height,image_width):
    
    tensor_input = np.array(features).reshape(-1,image_height,image_width)
    
    return tensor_input
	
def prepare_to_train(tensor_input,labels,test_size,random_state,num_classes):
    
    x_train,x_test,y_train,y_test = train_test_split(tensor_input,labels,test_size=test_size,random_state=random_state)
    
    y_train = to_categorical(y_train,num_classes=num_classes)
    y_test = to_categorical(y_test,num_classes=num_classes)
    
    return x_train,x_test,y_train,y_test


def build_conv_layer(filters,kernel_row,kernel_col,pooling_row,pooling_col,activation,dropout,padding,model):
    
    model.add(Conv2D(filters,kernel_size=(kernel_row,kernel_col),activation=activation,padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(filters,kernel_size=(kernel_row,kernel_col),activation=activation,padding=padding))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(pooling_row,pooling_col),padding=padding))
    model.add(Dropout(dropout))
    
    return model

def first_conv_layer(filters,kernel_row,kernel_col,pooling_row,pooling_col,activation,dropout,padding,image_height,image_width,model):
 
    model.add(Conv2D(filters,kernel_size=(kernel_row,kernel_col),activation=activation,input_shape=(input_height,input_width,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters,kernel_size=(kernel_row,kernel_col),activation=activation,padding=padding))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(pooling_row,pooling_col),padding=padding))
    model.add(Dropout(dropout))
    
    return model

def last_layer(model,activation,num_classes):
    
    model.add(Flatten())
    model.add(Dense(units=512,activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=128,activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(units=128,activation='softmax'))
    
    return model
