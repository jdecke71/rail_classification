

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import json
import sys
from datetime import datetime

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.metrics import r2_score


def WriteJSON(obj,filename):
    with open(filename, 'w+') as outfile:
        try:
            obj_json = json.dumps(obj, sort_keys=True, indent=4,default=str)
            outfile.write(obj_json)
        except Exception as e:
            print(e, file=sys.stderr)
            print('File not written.')


def ReadJSON(filename):
    obj = []
    try: 
        with open(filename, 'r') as infile:
            obj = json.load(infile)
    except Exception as e:
        print(e, file=sys.stderr)
        print('File not found.')
        
    return obj


def FitAndScoreCLA(df,params,features,labels,classifiers,testSize=0.20):
    print('classifying')

    X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size = testSize, random_state=42)
    
    clfs = []
    for classifier in classifiers:
        tmp = {}
        clf = classifier['Method']
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        # r2 = r2_score(y_true, y_pred)
        
        # Get report and matrix for display
        print('Classification report for  -',classifier['Name'])
        print('-----------------------------------------------------------------------------------------------')
        print(" %s:\n%s\n"% (clf, classification_report(y_test, y_pred)))
        
        tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
        print(classifier['Name'],'Confusion Matrix')
        print('   P0 \t P1 ')
        print('A0',tn,'\t',fp)
        print('A1',fn,'\t',tp)
        print('\n')
        
        
        # Get report and matrix for file
        clr = classification_report(y_test, y_pred,output_dict=True)
        cnm = list(confusion_matrix(y_test,y_pred))
        
        tmp[classifier['Name']] = {'Report':clr,
                                  'Matrix':cnm}
        clfs.append(tmp)  
        
    # Open results file, append new result, write to file
    resultsObj = ReadJSON(params['results_file'])
    
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    
    currResults = {'Description':params['description'],
                   'classifiers':clfs,
                   'Run Time':date_time,
                   'Sample Size':params['sample_size'],
                   'Image Resolution':params['img_size'],
                   'Counts':{'0':dict(df.Catenary.value_counts())[0],'1':dict(df.Catenary.value_counts())[1]},
              }
    
    resultsObj.append(currResults)
    WriteJSON(resultsObj,params['results_file'])


def DefineParameters():

    '''
    Parameters 
    ----------
    Set for each test. 


    img_folder: Root folder of image collection

    results_file: JSON file for output of results and metadata

    description: String for labeling/notes

    sample_size: Sample size to pull from each csv, 0-1

    img_size: Native resolution is 1280x1280

    '''

    img_folder = '../data/output_images/'

    results_file = '../data/results/'+'results2.json'

    # description = input('Enter Description: ')

    # sample_size = float(input('Enter sample size: '))

    # tmp = int(input('Enter single dimension for image size: '))
    img_size = (1280,1280)

    params = {}

    params['description'] = 'test'
    params['img_folder'] = img_folder
    params['results_file'] = results_file
    params['sample_size'] = 1.0
    params['img_size'] = img_size

    return params


def LoadCSVData(params):
    print('Loading CSVs')

    '''
    Loads csv only, no images.
    '''

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
            filename = params['img_folder']+key+'/'+value+'.csv'
            tmp = pd.read_csv(filename,header=0)
            tmp['Railway'] = key
            
            # Take sample from each folder 
            tmp = tmp.sample(frac=params['sample_size']).reset_index(drop=True)
            frames.append(tmp)
        except Exception as e:
            print(e)

    df = pd.concat(frames)

    df = df.dropna()
    df['Catenary'] = df['Catenary'].astype(int)

    return df



def LoadNonCat(data,params):
    print('Loading non-cat CSVs')

    '''
    Open known non-catenary lines and add differntial to df
    '''

    zeros = data.Catenary.value_counts()[0]
    ones = data.Catenary.value_counts()[1]

    names = [
        'Amtrak_non_cat_1',
        'Amtrak_non_cat_2',
        'Amtrak_non_cat_3'
    ]

    abbr = [
        'ANC',
        'ANC2',
        'ANC3'
    ]
    locations = dict(zip(names,abbr))

    diff = ones - zeros

    if diff > 0:
        frames = []
        for key,value in locations.items():
            try:
                filename = params['img_folder']+key+'/'+value+'.csv'
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
            data = pd.concat([data,duds]).reset_index(drop=True)
        except Exception as e:
            print(e)
            duds = duds.sample(len(duds.index.tolist())).reset_index(drop=True)
            data = pd.concat([data,duds]).reset_index(drop=True)

    return data
        


def LoadImages(frame,params):
    print('Loading Images')
    '''
    Load images into df
    '''
    rows = frame.index.tolist()
    # print(type(data))
    images = []
    for row in rows:
        img_path = params['img_folder']+frame.iloc[row]['Railway']+'/'+frame.iloc[row]['Name']+'.png'
        img = Image.open(img_path).convert('RGBA')
        img.thumbnail(params['img_size'], Image.ANTIALIAS)
        data = np.asarray(img)
        data = data.flatten()
        # Append img instead of data if you want as image       
        images.append(data)
        
    frame['Image'] = images

    cols = ['Catenary','Image']
    frame = frame[cols]

    return frame


def GetLabelsFeatures(data):
    print('Getting labels and features')
    labels = np.asarray(data.Catenary.tolist())
    features = np.asarray(data.Image.tolist())

    return labels,features


def DefineClassifiers():

    '''
    Setup classifiers
    '''

    BGN = {'Name':'BGN',
           'Method': GaussianNB()}

    DTC = {'Name':'DTC',
           'Method': DecisionTreeClassifier()}

    KNN = {'Name':'KNN',
           'Method': KNeighborsClassifier()}

    SVM = {'Name':'SVM',
           'Method': SVC(gamma=0.001)}


    classifiers = [BGN,DTC,KNN,SVM]

    return classifiers


def main():

    params = DefineParameters()

    df = LoadCSVData(params)

    df = LoadNonCat(df,params)
    
    # print(df.head())

    LoadImages(df,params)

    labels,features = GetLabelsFeatures(df)

    classifiers = DefineClassifiers()

    FitAndScoreCLA(df,params,features,labels,classifiers)

main()







