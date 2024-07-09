
import pandas as pd
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
tf.config.list_physical_devices('GPU')
# from tensorflow.keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, concatenate, Reshape,BatchNormalization, MultiHeadAttention, Concatenate, Dropout, Activation, Bidirectional, LSTM
import keras.backend as K
import tensorflow as tf
# from keras.layers.recurrent import SimpleRNN
# from keras import layers
from keras.optimizers import Adam
# from CapsuleLayer import Capsule
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
from bert import bert
import matplotlib
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
import tensorflow as tf
from sklearn.model_selection import train_test_split
from Capsule_MPNN import *
from ReadData import *
# from keras.optimizers.legacy import Adam
import tensorflow as tf
from sklearn.model_selection import train_test_split
# import pickle
def split(mirna, drug, y, drug_encoder, name,dti):
    if not os.path.exists(name):
        os.makedirs(name)

    if drug_encoder == "chemmolefusion":
        train_p, test_p, train_d, test_d, train_y, test_y = train_test_split(mirna, drug[0], y, test_size=0.2,
                                                                                random_state=1, stratify=y)  # 1
        with open(name + "/sample_summary.txt", "w") as fw:
            fw.write("train_mirna\ttest_mirna\ttrain_drug\ttest_drug\ttrain_y\ttest_y\n")
    train_y = to_categorical(train_y, num_classes=2)
    test_y = to_categorical(test_y, num_classes=2)
    return train_p, test_p, train_d, test_d, train_y, test_y

from tensorflow import keras
import tensorflow as tf

def custom_f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = (true_positives + K.epsilon()) / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = (true_positives + K.epsilon()) / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def model_bert_chemmolefusion_capsule(param):
    with tf.device('/gpu:0'):
        sequence_input_1 = Input(shape=768)
        model_p = Flatten()(sequence_input_1)
        model_p = Dense(param['target_dense'],kernel_regularizer=keras.regularizers.l2(0.01))(model_p)
        # model_p = Dropout(0.5)(model_p)
        model_p = BatchNormalization()(model_p)
        model_p = Activation('relu')(model_p)

        sequence_input_2 = Input(shape=684)
        model_d = Flatten()(sequence_input_2)
        model_d = Dense(param['target_dense'],kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
        # model_d = Dropout(0.5)(model_d)
        model_d = BatchNormalization()(model_d)
        model_d = Activation('relu')(model_d)

        model = concatenate([model_p, model_d])
        model = Reshape((-1, 8))(model)

        primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1,
                                 padding='valid')
        capsule = Capsule(num_capsule=param['num_capsule'], dim_capsule=16, routings=param['routings'],
                          share_weights=True)(
            primarycaps)

        # capsule = BPCapsNet(2,2)(model)

        length = Length()(capsule)
        # digit_caps_len = Length(name='length_capsnet_output')(digit_caps)

        model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=length)
        # model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=digit_caps_len)
        model.summary()

        return model
