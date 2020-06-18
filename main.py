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


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh



from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score




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

def preProcessFeautureVector(dataX_in):
    
    dataX_mean = dataX_in.mean(axis = 1)
    dataX_std = dataX_in.std(axis = 1)   
    dataX_mean6 =  pd.concat([dataX_mean]*len(gestures), axis = 1)
    dataX_std6 =  pd.concat([dataX_std]*len(gestures), axis = 1)   
    dataX6 = (dataX_in - dataX_mean6)/dataX_std6
    
    return dataX6


def trainFeedForwardNetwork(X_train,y_train):
    
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = None, input_dim = 6))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'tanh'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = len(X_train), validation_split= 0.3 ,epochs = 1000)
    
    return classifier





def classifyEMG_SegmentationNN(dataX_test, centers, model):
    
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
            
            filt_window_emg = window_emg.apply(preProcessEMGSegment)
            window_emg = filt_window_emg.loc[idx_start:idx_end]
            
            
            t_filt = time.time() - tStart
            
            tStart = time.time()
            featVector = featureExtraction([window_emg], centers)
            featVectorP = preProcessFeautureVector(featVector)
            t_featExtra =  time.time() - tStart
            
            tStart = time.time()
            x = model.predict(featVectorP).tolist()
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
        
        
    return vecTime, timeSeq, predLabel_seq   



def unique(list1): 
    # insert the list to the set 
    list_set = set(list1) 
    # convert the set to the list 
    unique_list = (list(list_set)) 
    
    return unique_list 


def posProcessLabels(predictions):
    
    predictions[0] = 1
    postProcessedLabels = predictions
        
    for i in range(1,len(predictions)):
        
        if predictions[i] == predictions[i-1]:
            cond = 1
        else:    
            cond = 0
            
        postProcessedLabels[i] =  (1 * cond) + (predictions[i]* (1 - cond))  
            
    uniqueLabels = unique(postProcessedLabels)
    
    an_iterator = filter(lambda number: number != 1, uniqueLabels)
    uniqueLabelsWithoutRest = list(an_iterator)
       
    if not uniqueLabelsWithoutRest:
        
        finalLabel = 1
        
    else:
        
        if len(uniqueLabelsWithoutRest) > 1:
            finalLabel = uniqueLabelsWithoutRest[0]
            
        else:
            finalLabel = uniqueLabelsWithoutRest[0]
                   
    
    return finalLabel





dataY = list(itertools.chain.from_iterable(itertools.repeat(x, 25) for x in range(1,len(gestures)+1)))






y_train = np.array(dataY)
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
dummy_y = np_utils.to_categorical(encoded_Y)
estimator = trainFeedForwardNetwork(X_train,dummy_y)



responses_label = []
predict_vector = []


test_FilteredX = []

test_samples = user['testingSamples']

sample = test_samples['pinch']['sample5']['emg']
df_test = pd.DataFrame.from_dict(sample)
# df = df_test.apply(preProcessEMGSegment)

# if segmentation == True:
#     df_sum  = df.sum(axis=1)
#     idx_Start, idx_End = detectMuscleActivity(df_sum)
# else:
#     idx_Start = 0;
#     idx_End = len(df)
    
# df_seg = df.iloc[idx_Start:idx_End]   
# test_FilteredX.append(df_seg)

vec_time, time_seq, prediq_seq = classifyEMG_SegmentationNN(df_test, centers, estimator)

predicted_label = posProcessLabels(prediq_seq)

print(predicted_label)
responses_label.append(predicted_label)
predict_vector.append(prediq_seq) 




# for move in gestures:   
#     for i in range(1,6):
#         sample = test_samples[move]['sample%s' %i]['emg']
#         df_test = pd.DataFrame.from_dict(sample)
#         # df = df_test.apply(preProcessEMGSegment)
        
#         # if segmentation == True:
#         #     df_sum  = df.sum(axis=1)
#         #     idx_Start, idx_End = detectMuscleActivity(df_sum)
#         # else:
#         #     idx_Start = 0;
#         #     idx_End = len(df)
            
#         # df_seg = df.iloc[idx_Start:idx_End]   
#         # test_FilteredX.append(df_seg)
    
#         vec_time, time_seq, prediq_seq = classifyEMG_SegmentationNN(df_test, centers, estimator)
    
#         predicted_label = posProcessLabels(prediq_seq)
        
#         print(predicted_label)
#         responses_label.append(predicted_label)
#         predict_vector.append(prediq_seq) 
        
        
        
    
# X_te = featureExtraction(test_FilteredX, centers) 
# X_test = preProcessFeautureVector(X_te)

# results = estimator.predict(X_test).tolist()   

# res = []

# for item in results: 
#     max_probNN = max(item)
#     predicted_labelNN = item.index(max_probNN) + 1
#     res.append(predicted_labelNN)


# cm = confusion_matrix(dataY, res)
# f = sns.heatmap(cm, annot=True)


# score =  accuracy_score(dataY, res) 

# percentage = "{:.2%}".format(score)
# print(percentage)

















