#
#https://github.com/RamiHaf/Explainable-Federated-Learning-via-Random-Forests/blob/main/Explaining_the_prediction_via_Feature_importances%20.ipynb
#
import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as kr
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
import os
import csv

from scipy.spatial.distance import euclidean
from sklearn.metrics import confusion_matrix

from time import sleep
from tqdm import tqdm

import copy
import numpy
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from numpy.random import RandomState
import scipy as scp
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils import to_categorical
from keras import backend as K
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from sklearn import mixture

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#%matplotlib inline


# Enter here the numb er of peers you want in the experiments
n_peers = 100
# the random state we will use in the experiments. It can be changed
rs = RandomState(42)

# Number of global training epochs
n_rounds = 10
# Number of local training epochs per global training epoch
n_local_rounds = 5

# Local batch size
local_batch_size = 32

# Global learning rate or 'gain'
model_substitution_rate = 1.0

# Clear nans and infinites in model updates
clear_nans = True

# the dectinary
FI_dic1= {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

# select aa random peer to be the scanner peer
peers_selected=random.sample(range(n_peers), 1)
scaner = peers_selected[0]

# Percentage and number of peers participating at each global training epoch
percentage_participants = 1.0
n_participants = int(n_peers * percentage_participants)

###########
# METRICS #
###########
metrics = {'accuracy': [],
          'atk_effectivity': [],
          'update_distances': [],
          'outliers_detected': [],
          'acc_no_target': []}

data = pd.read_csv('IoTID20_preprocessada.csv')
# Drop all records with missing values
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
data['Label'] = data['Label'].astype('category')
data = data.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(data[:,:-1],data[:,-1:],test_size = 0.20,random_state = rs )
names = ['Protocol','Flow_Duration','TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Flow_Byts/s','Flow_IAT_Std','Fwd_IAT_Tot','Fwd_Header_Len','Bwd_Header_Len','Pkt_Len_Std','ACK_Flag_Cnt','Init_Bwd_Win_Byts']
Features_number = len(X_train[0])


earlystopping = EarlyStopping(monitor = 'val_loss',
                              min_delta = 0.01,
                              patience = 50,
                              verbose = 1,
                              baseline = 2,
                              restore_best_weights = True)

checkpoint = ModelCheckpoint('test.h8',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

model = Sequential()
model.add(Dense(70, input_dim=Features_number, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.add(Flatten())
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

#print('X_train:',np.shape(X_train))
#print('y_train:',np.shape(y_train))
#print('X_test:',np.shape(X_test))
#print('y_test:',np.shape(y_test))

history = model.fit(X_train, y_train,epochs=20,validation_data=(X_test, y_test),callbacks = [checkpoint, earlystopping],shuffle=True)
#history = model.fit(X_train, y_train,epochs=2,validation_data=(X_test, y_test),callbacks = [checkpoint, earlystopping],shuffle=True)

#AUXILIARY METHODS FOR FEDERATED LEARNING

# RETURN INDICES TO LAYERS WITH WEIGHTS AND BIASES
def trainable_layers(model):
    return [i for i, layer in enumerate(model.layers) if len(layer.get_weights()) > 0]

# RETURN WEIGHTS AND BIASES OF A MODEL
def get_parameters(model):
    weights = []
    biases = []
    index = trainable_layers(model)
    for i in index:
        weights.append(copy.deepcopy(model.layers[i].get_weights()[0]))
        biases.append(copy.deepcopy(model.layers[i].get_weights()[1]))

    return weights, biases

# SET WEIGHTS AND BIASES OF A MODEL
def set_parameters(model, weights, biases):
    index = trainable_layers(model)
    for i, j in enumerate(index):
        model.layers[j].set_weights([weights[i], biases[i]])

# DEPRECATED: RETURN THE GRADIENTS OF THE MODEL AFTER AN UPDATE
def get_gradients(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)

    w_grad = [w for i,w in enumerate(output_grad) if i%2==0]
    b_grad = [w for i,w in enumerate(output_grad) if i%2==1]

    return w_grad, b_grad

# RETURN THE DIFFERENCE OF MODELS' WEIGHTS AND BIASES AFTER AN UPDATE
# NOTE: LEARNING RATE IS APPLIED, SO THE UPDATE IS DIFFERENT FROM THE
# GRADIENTS. IN CASE VANILLA SGD IS USED, THE GRADIENTS ARE OBTAINED
# AS (UPDATES / LEARNING_RATE)
def get_updates(model, inputs, outputs, batch_size, epochs):
    w, b = get_parameters(model)
    #model.train_on_batch(inputs, outputs)
    model.fit(inputs, outputs, batch_size=batch_size, epochs=epochs, verbose=0)
    w_new, b_new = get_parameters(model)

    weight_updates = [old - new for old,new in zip(w, w_new)]
    bias_updates = [old - new for old,new in zip(b, b_new)]

    return weight_updates, bias_updates

# UPDATE THE MODEL'S WEIGHTS AND PARAMETERS WITH AN UPDATE
def apply_updates(model, eta, w_new, b_new):
    w, b = get_parameters(model)
    new_weights = [theta - eta*delta for theta,delta in zip(w, w_new)]
    new_biases = [theta - eta*delta for theta,delta in zip(b, b_new)]
    set_parameters(model, new_weights, new_biases)

# FEDERATED AGGREGATION FUNCTION
def aggregate(n_layers, n_peers, f, w_updates, b_updates):
    agg_w = [f([w_updates[j][i] for j in range(n_peers)], axis=0) for i in range(n_layers)]
    agg_b = [f([b_updates[j][i] for j in range(n_peers)], axis=0) for i in range(n_layers)]
    return agg_w, agg_b

# SOLVE NANS
def nans_to_zero(W, B):
    W0 = [np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0) for w in W]
    B0 = [np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0) for b in B]
    return W0, B0

def build_forest(X,y):
    clf=RandomForestClassifier(n_estimators=1000, max_depth=7, random_state=42, verbose = 1)
    clf.fit(X,y)
    return clf

# COMPUTE EUCLIDEAN DISTANCE OF WEIGHTS
def dist_weights(w_a, w_b):
    wf_a = flatten_weights(w_a)
    wf_b = flatten_weights(w_b)
    return euclidean(wf_a, wf_b)

# TRANSFORM ALL WEIGHT TENSORS TO 1D ARRAY
def flatten_weights(w_in):
    h = w_in[0].reshape(-1)
    for w in w_in[1:]:
        h = np.append(h, w.reshape(-1))
    return h

# scan the forest for trees maches the wrong predictions of the black-box
def scan_wrong(forest_predictions, FL_predict1, forest , y_test_local, X_test_local):
    sum_feature_improtance= 0
    overal_wrong_feature_importance = 0
    counter = 0
    second_counter = 0
    never_seen = 0
    avr_wrong_importance = 0
    FL_predict1 = np.argmax(FL_predict1, axis=1)
    forest_predictions = np.argmax(forest_predictions, axis=0)
    y_test_local = np.argmax(y_test_local, axis=1)
    for i in range (len(FL_predict1)):
        i_tree = 0
#         if the black-box got a wrong prediction
        if (FL_predict1[i] != y_test_local[i]):
#         getting the prediction of the trees one by one
            for tree_in_forest in forest.estimators_:
                sample = X_test_local[i].reshape(1, -1)
                temp = forest.estimators_[i_tree].predict(sample)
#                temp =  np.argmax(temp, axis=1)
                temp =  np.argmax(temp, axis=0)                
                i_tree = i_tree + 1
#  if the prediction of the tree maches the predictions of the black-box
                if(FL_predict1[i] == temp):
#         getting the features importances
                    sum_feature_improtance = sum_feature_improtance + tree_in_forest.feature_importances_
                    counter = counter + 1
#         if we have trees maches the black-box predictions
        if(counter>0):
            ave_feature_importence = sum_feature_improtance/counter
            overal_wrong_feature_importance = ave_feature_importence + overal_wrong_feature_importance
            second_counter = second_counter + 1
            counter = 0
            sum_feature_improtance = 0
#             if there is no trees maches the black-box predictions
        else:
            if(FL_predict1[i] != y_test_local[i]):
                never_seen = never_seen +1
#                 getting the average features importances for all the samples that had wrong predictions.
    if(second_counter>0):
        avr_wrong_importance = overal_wrong_feature_importance / second_counter
    return forest.feature_importances_

trainable_layers(model)
get_parameters(model)
get_updates(model, X_train, y_train, 32, 2)

W = get_parameters(model)[0]
B = get_parameters(model)[1]

# BASELINE SCENARIO
#buid the model as base line for the shards (sequential)
# Number of peers
#accordin to what we need
ss = int(len(X_train)/n_peers)
inputs_in = X_train[0*ss:0*ss+ss]
outputs_in = y_train[0*ss:0*ss+ss]
def build_model(X_t, y_t):
    model = Sequential()
    model.add(Dense(70, input_dim=Features_number, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))    
    model.add(Flatten())    
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.fit(X_t,
              y_t,        
              batch_size=32, 
              epochs=250, 
              verbose=1,
              validation_data=((X_test, y_test)))
    return model

print(model.summary())

# predict probabilities for test set
yhat_probs = model.predict(X_test, verbose=0)
# predict crisp classes for test set

#yhat_classes = model.predict .predict_classes(X_test, verbose=0)
yhat_classes = np.argmax(yhat_probs, axis=1)


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
print('F1 score: %f' % f1)


# confusion matrix
#mat = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
#
#display(mat)
#plt.matshow(mat);
#plt.colorbar()
#plt.show()

number_for_threshold1 = numpy.empty(20, dtype=float)
number_for_threshold2 = numpy.empty(20, dtype=float)
for r in range(len(number_for_threshold1)):
    number_for_threshold1[r] = 0
    number_for_threshold2[r] = 0

####################################
# MODEL AND NETWORK INITIALIZATION #
####################################
inputs = X_train[0*ss:0*ss+ss]
outputs = y_train[0*ss:0*ss+ss]
global_model = build_model(inputs,outputs)
n_layers = len(trainable_layers(global_model))

print('Initializing network.')
network = []
for i in tqdm(range(n_peers)):
    ss = int(len(X_train)/n_peers)
    inputs = X_train[i*ss:i*ss+ss]
    outputs = y_train[i*ss:i*ss+ss]
    network.append(global_model)

##################
# BEGIN TRAINING #
##################
for t in range(n_rounds):
    print(f'Round {t+1}.')

    ## SERVER SIDE #################################################################
    # Fetch global model parameters
    global_weights, global_biases = get_parameters(global_model)

    if clear_nans:
        global_weights, global_biases = nans_to_zero(global_weights, global_biases)

    # Initialize peer update lists
    network_weight_updates = []
    network_bias_updates = []

    # Selection of participant peers in this global training epoch
    participants = random.sample(list(enumerate(network)),n_participants)
    ################################################################################


    ## CLIENT SIDE #################################################################
    for i, local_model in tqdm(participants):

        # Update local model with global parameters
        set_parameters(local_model, global_weights, global_biases)

        # Initialization of user data
        ss = int(len(X_train)/n_peers)
        inputs = X_train[i*ss:i*ss+ss]
        outputs = y_train[i*ss:i*ss+ss]

# the scanner peer side
        if(i == scaner):
            X_train_local, X_test_local, y_train_local, y_test_local = train_test_split(inputs,outputs, test_size=0.7, random_state=rs)
            inputs = X_train_local
            outputs = y_train_local
            if(t == 0):
                forest = build_forest(X_train_local,y_train_local)
            forest_predictions = forest.predict(X_test_local)
            acc_forest = np.mean([t==p for t,p in zip(y_test_local, forest_predictions)])
            FL_predict1 = global_model.predict(X_test_local)
            imp = scan_wrong(forest_predictions, FL_predict1, forest , y_test_local, X_test_local)
            FI_dic1[t] = imp


 # Benign peer
                # Train local model
        local_weight_updates, local_bias_updates = get_updates(local_model,
                                                                       inputs, outputs,
                                                                       local_batch_size, n_local_rounds)
        if clear_nans:
            local_weight_updates, local_bias_updates = nans_to_zero(local_weight_updates, local_bias_updates)
        network_weight_updates.append(local_weight_updates)
        network_bias_updates.append(local_bias_updates)

    ## END OF CLIENT SIDE ##########################################################

    ######################################
    # SERVER SIDE AGGREGATION MECHANISMS #
    ######################################


        # Aggregate client updates
    aggregated_weights, aggregated_biases = aggregate(n_layers,
                                                      n_participants,
                                                      np.mean,
                                                      network_weight_updates,
                                                      network_bias_updates)

    if clear_nans:
        aggregated_weights, aggregated_biases = nans_to_zero(aggregated_weights, aggregated_biases)

    # Apply updates to global model
    apply_updates(global_model, model_substitution_rate, aggregated_weights, aggregated_biases)

    # Proceed as in first case
    aggregated_weights, aggregated_biases = aggregate(n_layers,
                                                      n_participants,
                                                      np.mean,
                                                      network_weight_updates,
                                                      network_bias_updates)
    if clear_nans:
        aggregated_weights, aggregated_biases = nans_to_zero(aggregated_weights, aggregated_biases)

    apply_updates(global_model, model_substitution_rate, aggregated_weights, aggregated_biases)

    ###################
    # COMPUTE METRICS #
    ###################

    # Global model accuracy
    score = global_model.evaluate(X_test, y_test, verbose=0)
    print(f'Global model loss: {score[0]}; global model accuracy: {score[1]}')
    metrics['accuracy'].append(score[1])


    # Accuracy without the target
    score = global_model.evaluate(X_test, y_test, verbose=0)
    metrics['acc_no_target'].append(score[1])


    # Distance of individual updates to the final aggregation
    metrics['update_distances'].append([dist_weights(aggregated_weights, w_i) for w_i in network_weight_updates])


# sort the feature according to the last epoch and print it with importances

print(np.size(FI_dic1[9]))
sort_index = np.argsort(FI_dic1[9])
for x in sort_index:
    print(names[x], ', ', FI_dic1[9][x])
