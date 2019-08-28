
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split


img_folder = '../data/output_images/'

'''
Loads csv only, no images.
'''
def GetCSVs(sample_size):

    # Name of folder
    names = [
        'Australia',
        'China',
        'Germany',
        'NewarkLR',
        'Switzerland',
        'Amtrak',
        'BostonMTBA',
        'DenverRTD',
        'LosAngelesMR',
        'SeattleLLR',
        'Netherlands'
    ]

    # Name of csv
    abbr = [
        'AUS',
        'CHN',
        'GRM',
        'NEW',
        'SWZ',
        'AMT',
        'BOS',
        'DEN',
        'LAA',
        'SEA',
        'NET'
    ]
    locations = dict(zip(names,abbr))

    # Collect each csv into one df adding railway name
    frames = []
    for key,value in locations.items():
        try:
            filename = img_folder+key+'/'+value+'.csv'
            tmp = pd.read_csv(filename,header=0)
            tmp['Railway'] = key

            # Take sample from each folder 
            tmp = tmp.sample(frac=sample_size).reset_index(drop=True)
            frames.append(tmp)
        except Exception as e:
            print(e)

    df = pd.concat(frames)

    df = df.dropna()
    df['Catenary'] = df['Catenary'].astype(int)
    
    
    '''
    Open known non-catenary lines and add differntial to df
    '''


    zeros = df.Catenary.value_counts()[0]
    ones = df.Catenary.value_counts()[1]

    names = [
        'Amtrak_non_cat_1',
        'Amtrak_non_cat_2',
        'Amtrak_non_cat_3',
        'Random'
    ]

    abbr = [
        'ANC',
        'ANC2',
        'ANC3',
        'RAN'
    ]

    locations['Amtrak_non_cat_1'] = 'ANC'
    locations['Amtrak_non_cat_2'] = 'ANC2'
    locations['Amtrak_non_cat_3'] = 'ANC3'
    locations['Random'] = 'RAN'

    locations2 = dict(zip(names,abbr))

    diff = ones - zeros

    if diff > 0:
        frames = []
        for key,value in locations2.items():
            try:
                filename = img_folder+key+'/'+value+'.csv'
                tmp = pd.read_csv(filename,header=0)
                tmp['Railway'] = key
                frames.append(tmp)
            except Exception as e:
                print(e)

        try:
            duds = pd.concat(frames)
            duds = duds.dropna()
            duds['Catenary'] = duds['Catenary'].astype(int) 

            duds = duds.sample(n=diff).reset_index(drop=True)
            df = pd.concat([df,duds]).reset_index(drop=True)
        except Exception as e:
            print(e)
            duds = duds.sample(len(duds.index.tolist())).reset_index(drop=True)
            df = pd.concat([df,duds]).reset_index(drop=True)
            
        return df


# In[6]:


'''
Get image paths and labels as lists
'''
def GetPaths(df):

    rows = df.index.tolist()
    path = GetABSPath(img_folder)
    img_paths = []
    labels = []
    for row in rows:
        tmp = df.iloc[row]
        img_path = path+'/'+tmp.Railway+'/set_2/'+tmp.Name+'.png'
        img_paths.append(img_path)
        label = int(tmp.Catenary)
        labels.append(label)

    print(len(img_paths))
    
    return img_paths,labels


# In[7]:


def GetABSPath(folder):
    return os.path.abspath(folder)


# In[8]:
def DataAugment(image):
    image = tf.keras.preprocessing.image.random_rotation(
        image,
        30,
        row_axis=1,
        col_axis=2,
        channel_axis=0,
        fill_mode='nearest',
        cval=0.0,
        interpolation_order=1)

    return image


def PreprocessImage(img_path):
    img_raw = tf.io.read_file(img_path)
    image = tf.io.decode_png(img_raw, channels=3)
    image = tf.image.resize(image, img_size)
    # print(image.shape)
    image /= 255.0  # normalize to [0,1] range

    return image

def flip(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x

def rotate(x: tf.Tensor) -> tf.Tensor:
    return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


def SplitDataSet(img_paths, labels):
    
    # split lists into training/test    
    X_train, X_test, y_train, y_test = train_test_split(img_paths,labels,test_size = .1, random_state=1)

    # split lists into training/validation   
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size = .2, random_state=1)

    print('Number of images in train: ', len(X_train))
    print("Distribution for train set: ", np.unique(y_train, return_counts=True))
    print('\n')

    print('Number of images in validation: ', len(X_val))
    print("Distribution for validation set: ", np.unique(y_val, return_counts=True))
    print('\n')

    print('Number of images in test: ', len(X_test))
    print("Distribution for test set: ", np.unique(y_test, return_counts=True))
    print('\n')

    # -----------------------------------
    # train
    # Read images/labels into tensor data    
    train_path_ds = tf.data.Dataset.from_tensor_slices(X_train)
    train_image_ds = train_path_ds.map(PreprocessImage, num_parallel_calls=AUTOTUNE)

    augmentations = [flip,rotate]
    for f in augmentations:
        train_image_ds = train_image_ds.map(lambda x: tf.cond(tf.random_uniform([], 0, 1) > 0.75, lambda: f(x), lambda: x), num_parallel_calls=4)
    train_image_ds = train_image_ds.map(lambda x: tf.clip_by_value(x, 0, 1))

    train_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(y_train, tf.int64))
    
    # Combine into dataset     
    train_image_label_ds = tf.data.Dataset.zip((train_image_ds, train_label_ds))


    # -----------------------------------
    # validation
    # Read images/labels into tensor data    
    val_path_ds = tf.data.Dataset.from_tensor_slices(X_val)
    val_image_ds = val_path_ds.map(PreprocessImage, num_parallel_calls=AUTOTUNE)
    augmentations = [flip,rotate]
    for f in augmentations:
        val_image_ds = val_image_ds.map(lambda x: tf.cond(tf.random_uniform([], 0, 1) > 0.75, lambda: f(x), lambda: x), num_parallel_calls=4)
    val_image_ds = val_image_ds.map(lambda x: tf.clip_by_value(x, 0, 1))
    val_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(y_val, tf.int64))
    
    # Combine into dataset     
    val_image_label_ds = tf.data.Dataset.zip((val_image_ds, val_label_ds))
    
    
    # -----------------------------------
    # test
    test_path_ds = tf.data.Dataset.from_tensor_slices(X_test)
    test_image_ds = test_path_ds.map(PreprocessImage, num_parallel_calls=AUTOTUNE)
    augmentations = [flip,rotate]
    for f in augmentations:
        test_image_ds = test_image_ds.map(lambda x: tf.cond(tf.random_uniform([], 0, 1) > 0.75, lambda: f(x), lambda: x), num_parallel_calls=4)
    test_image_ds = test_image_ds.map(lambda x: tf.clip_by_value(x, 0, 1))
    test_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(y_test, tf.int64))
    
    test_image_label_ds = tf.data.Dataset.zip((test_image_ds, test_label_ds))
    
    return train_image_label_ds, val_image_label_ds, test_image_label_ds


# In[10]:


'''
Shuffle/batch/prefetch/Set Range
'''
def ShuffleBatch(ds_dict,buff,BATCH_SIZE = 32):
    
    ds = ds_dict.shuffle(buffer_size = buff)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(buffer_size=AUTOTUNE)
    # ds

    def change_range(image,label):
        return 2*image-1, label

    keras_ds = ds.map(change_range)
    
    return keras_ds


