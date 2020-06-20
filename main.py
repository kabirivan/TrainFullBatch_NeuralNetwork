#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 21:06:52 2020

@author: aguasharo
"""
from __future__ import print_function

import json
import os
import itertools
import pandas as pd
import math
from scipy import signal
import numpy as np
import simplespectral as sp
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.manifold import TSNE
import seaborn as sns

import time


import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.optimizers import Adam, SGD

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh



from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


import random
from sklearn.preprocessing import StandardScaler



from collections import Counter


folderData = 'trainingJSON'
gestures = ['noGesture', 'fist', 'waveIn', 'waveOut', 'open', 'pinch']
#gestures = ['noGesture', 'fist']



files = []

for root, dirs, files in os.walk(folderData):
     print('Dataset Ready !')
         
         
file_selected = root + '/' + files[2]  

with open(file_selected) as file:
    user = json.load(file)     

# Training Process
train_samples = user['trainingSamples']





def butter_lowpass_filter(data, fs, order):
    # Get the filter coefficients 
    b, a = signal.butter(order, fs, 'low', analog = False)
    y = signal.filtfilt(b, a, data)
    return y


def preProcessEMGSegment(EMGsegment_in):
    
    EMG = max(EMGsegment_in)
    
    if EMG > 1:
        EMGnormalized = EMGsegment_in/128
    else:
        EMGnormalized = EMGsegment_in    
             
    EMGrectified = abs(EMGnormalized)   
    EMGsegment_out = butter_lowpass_filter(EMGrectified, 0.1, 5)
    
    
    return EMGsegment_out



def detectMuscleActivity(emg_sum):

    fs = 200
    minWindowLength_Segmentation =  100
    hammingWdw_Length = np.hamming(25)
    numSamples_lapBetweenWdws = 10
    threshForSum_AlongFreqInSpec = 0.857

    [s, f, t, im] = plt.specgram(emg_sum, NFFT = 25, Fs = fs, window = hammingWdw_Length, noverlap = numSamples_lapBetweenWdws, mode = 'magnitude', pad_to = 50)  
    sumAlongFreq = [sum(x) for x in zip(*s)]

    greaterThanThresh = []
    # Thresholding the sum sumAlongFreq
    for item in sumAlongFreq:
        if item >= threshForSum_AlongFreqInSpec:
            greaterThanThresh.append(1)
        else:
            greaterThanThresh.append(0)
           
    greaterThanThresh.insert(0,0)       
    greaterThanThresh.append(0)    
    diffGreaterThanThresh = abs(np.diff(greaterThanThresh)) 

    if diffGreaterThanThresh[-1] == 1:
        diffGreaterThanThresh[-2] = 1;      
       
    x = diffGreaterThanThresh[0:-1];
    findNumber = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    idxNonZero = findNumber(1,x)
    numIdxNonZero = len(idxNonZero)
    idx_Samples = np.floor(fs*t)

    if numIdxNonZero == 0:
        idx_Start = 1
        idx_End = len(emg_sum)
    elif numIdxNonZero == 1:
        idx_Start = idx_Samples[idxNonZero]
        idx_End = len(emg_sum)
    else:
        idx_Start = idx_Samples[idxNonZero[0]]
        idx_End = idx_Samples[idxNonZero[-1]-1]

    numExtraSamples = 25
    idx_Start = max(1,idx_Start - numExtraSamples)
    idx_End = min(len(emg_sum), idx_End + numExtraSamples)
    
    if (idx_End - idx_Start) < minWindowLength_Segmentation:
        idx_Start = 1
        idx_End = len(emg_sum)


    return int(idx_Start), int(idx_End)


def findCentersClass(emg_filtered,sample):
    distances = []
    column = np.arange(0,sample)
    #column = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
    mtx_distances = pd.DataFrame(columns = column)
    mtx_distances = mtx_distances.fillna(0) # with 0s rather than NaNs
    
    
    for sample_i in emg_filtered:
        for sample_j in emg_filtered:   
            dist, dummy = fastdtw(sample_i, sample_j, dist = euclidean)
            distances.append(dist)
            
        df_length = len(mtx_distances)
        mtx_distances.loc[df_length] = distances 
        distances= []  
    vector_dist = mtx_distances.sum(axis=0)
    idx = vector_dist.idxmin()
    center_idx = emg_filtered[int(idx)]
    
    return center_idx


def featureExtraction(emg_filtered, centers):

    dist_features = []
    
    column = np.arange(0,len(centers))
    dataX = pd.DataFrame(columns = column)
    dataX = dataX.fillna(0)
    
    for rep in emg_filtered:
        for middle in centers:
            dist, dummy = fastdtw(rep, middle, dist = euclidean) 
            dist_features.append(dist)
        
        dataX_length = len(dataX)
        dataX.loc[dataX_length] = dist_features
        dist_features = [] 
    
    return dataX

def preProcessFeatureVector(dataX_in):
    
    dataX_mean = dataX_in.mean(axis = 1)
    dataX_std = dataX_in.std(axis = 1)   
    dataX_mean6 =  pd.concat([dataX_mean]*len(gestures), axis = 1)
    dataX_std6 =  pd.concat([dataX_std]*len(gestures), axis = 1)   
    dataX6 = (dataX_in - dataX_mean6)/dataX_std6
    
    return dataX6


def trainFeedForwardNetwork(X_train,y_train, X_test, y_test):
    
    classifier = Sequential()
    
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = None, input_dim = 6))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'tanh'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = 150, epochs = 1200, validation_data = (X_test, y_test),verbose = 0 )
    
    return classifier




def classifyEMG_SegmentationNN(dataX_test, centers, model):
    sc = StandardScaler()
    window_length = 500
    stride_length = 10
    emg_length = len(dataX_test)
    predLabel_seq = []
    vecTime = []
    timeSeq = []
    
    
    count = 0
    while True:
        start_point = stride_length*count + 1
        end_point = start_point + window_length - 1
        
        if end_point > emg_length:
            break
        
        tStart = time.time()
        window_emg = dataX_test.iloc[start_point:end_point]   
        filt_window_emg = window_emg.apply(preProcessEMGSegment)
        window_sum  = filt_window_emg.sum(axis=1)
        idx_start, idx_end = detectMuscleActivity(window_sum)
        t_acq = time.time()-tStart
        
        if idx_start != 1 & idx_end != len(window_emg):
            
            tStart = time.time()
            
            filt_window_emg1 = window_emg.apply(preProcessEMGSegment)
            window_emg1 = filt_window_emg1.iloc[idx_start:idx_end]
            
            
            t_filt = time.time() - tStart
            
            tStart = time.time()
            featVector = featureExtraction([window_emg1], centers)
            featVectorP = preProcessFeatureVector(featVector)
            t_featExtra =  time.time() - tStart
            
            tStart = time.time()
            x = model.predict_proba(featVectorP).tolist()
            probNN = x[0]
            max_probNN = max(probNN)
            predicted_labelNN = probNN.index(max_probNN) + 1
            t_classiNN = time.time() - tStart
            
            tStart = time.time()
            if max_probNN <= 0.5:
                predicted_labelNN = 1
            t_threshNN = time.time() - tStart 
            #print(predicted_labelNN)
           
        else:
            
            t_filt = 0
            t_featExtra = 0
            t_classiNN = 0
            t_threshNN = 0
            predicted_labelNN = 1
            #print('1')
            
            
        count = count + 1
        predLabel_seq.append(predicted_labelNN)
        vecTime.append(start_point)
        timeSeq.append(t_acq + t_filt + t_featExtra + t_classiNN + t_threshNN)    
        
        
    return  predLabel_seq, vecTime, timeSeq



def unique(list1): 
    # insert the list to the set 
    list_set = set(list1) 
    # convert the set to the list 
    unique_list = (list(list_set)) 
    
    return unique_list 



def majorite_vote(data, before, after):
    
    votes =[0,0,0,0,0,0]
    class_maj = []
        
    for j in range(0,len(data)):
        wind_mv = data[max(0,(j-before)):min(len(data),(j+after))]
        
        for k in range(0, len(gestures)):
            a = [1 if i == k+1 else 0 for i in wind_mv]  
            votes[k] = sum(a)
            
        findNumber = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
        idx_label = findNumber(max(votes),votes)
        class_maj.append( idx_label[0] + 1)
        
    
    return class_maj





def post_ProcessLabels(predicted_Seq):
    
    vec = predicted_Seq.copy()
    pred = majorite_vote(vec, 4, 4)
    predictions = pred.copy()
    predictions[0] = 1
    postProcessed_Labels = predictions.copy()
        
    for i in range(1,len(predictions)):
        
        if predictions[i] == predictions[i-1]:
            cond = 1
        else:    
            cond = 0
            
        postProcessed_Labels[i] =  (1 * cond) + (predictions[i]* (1 - cond))  
            
    uniqueLabels = unique(postProcessed_Labels)
    
    an_iterator = filter(lambda number: number != 1, uniqueLabels)
    uniqueLabelsWithoutRest = list(an_iterator)
       
    if not uniqueLabelsWithoutRest:
        
        finalLabel = 1
        
    else:
        
        if len(uniqueLabelsWithoutRest) > 1:
            finalLabel = uniqueLabelsWithoutRest[0]
            
        else:
            finalLabel = uniqueLabelsWithoutRest[0]
                   
    
    return finalLabel, pred









dataY = list(itertools.chain.from_iterable(itertools.repeat(x, 25) for x in range(1,len(gestures)+1)))


center1 = pd.read_csv("center1.csv")
center2 = pd.read_csv("center2.csv")
center3 = pd.read_csv("center3.csv")
center4 = pd.read_csv("center4.csv")
center5 = pd.read_csv("center5.csv")
center6 = pd.read_csv("center6.csv")
X_train = pd.read_csv("X_train.csv")



center1 = center1.iloc[:,1:9] 
center2 = center2.iloc[:,1:9] 
center3 = center3.iloc[:,1:9] 
center4 = center4.iloc[:,1:9] 
center5 = center5.iloc[:,1:9] 
center6 = center6.iloc[:,1:9]


centers = [center1, center2, center3, center4, center5, center6] 
X_train1 = X_train.iloc[:,1:9]
y_train = np.array(dataY) 

dataX = X_train.iloc[:,1:9]
dataX['6'] = y_train

Xx_train = dataX.sample(frac=1).reset_index(drop=True)
Xx_train1 = Xx_train.iloc[:,0:6] 


Yy_train = Xx_train['6']






encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
dummy_y = np_utils.to_categorical(encoded_Y)


encoder = LabelEncoder()
encoder.fit(Yy_train)
encoded_Y = encoder.transform(Yy_train)
Yy_train = np_utils.to_categorical(encoded_Y)


estimator = trainFeedForwardNetwork(X_train1, dummy_y, Xx_train1, Yy_train)




responses_label = []
predict_vector = []

#%% Samples

for i in range(1,26):
    test_samples = user['testingSamples']
    sample = test_samples['fist']['sample%s' %i]['emg']
    df_test = pd.DataFrame.from_dict(sample)
    
    [predictedSeq, vec_time, time_seq]= classifyEMG_SegmentationNN(df_test, centers, estimator)
    #print("Before: ", predictedSeq)
       
    # class_post = majorite_vote(predictedSeq, 4, 4)
                     
    #print("Post: ", class_post)        
    predicted_label, post_Seq = post_ProcessLabels(predictedSeq)
    
    print("After: ", post_Seq)

    print("Result: ", predicted_label)
    #responses_label.append(predicted_label)
    #predict_vector.append(prediq_seq) 
    
    
    
    
    
    

#%%



# test_FilteredX = []
# test_samples = user['testingSamples']
# segmentation = True

# for move in gestures:   
#     for i in range(1,11):
#         sample = test_samples[move]['sample%s' %i]['emg']
#         df_test = pd.DataFrame.from_dict(sample)
#         df = df_test.apply(preProcessEMGSegment)
        
#         if segmentation == True:
#             df_sum  = df.sum(axis=1)
#             idx_Start, idx_End = detectMuscleActivity(df_sum)
#         else:
#             idx_Start = 0;
#             idx_End = len(df)
            
#         df_seg = df.iloc[idx_Start:idx_End]   
#         test_FilteredX.append(df_seg)
            
        
 
# data_Y = list(itertools.chain.from_iterable(itertools.repeat(x, 10) for x in range(1,len(gestures)+1)))
# y_test = np.array(data_Y)        
 
# X_te = featureExtraction(test_FilteredX, centers) 
# X_test = preProcessFeatureVector(X_te)

# results = estimator.predict(X_test).tolist()   

# res = []

# for item in results: 
#     max_probNN = max(item)
#     predicted_labelNN = item.index(max_probNN) + 1
#     res.append(predicted_labelNN)



# score =  accuracy_score(y_test, res) 

# percentage = "{:.2%}".format(score)
# print(percentage)

















