from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os

def prepare_data(data_file_name):
    """
Prepare Data for Neural Network Injection
    """
    header = ['class','cap-shape', 'cap-surface',
              'cap-color', 'bruises', 'odor', 'gill-attachment',
              'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
              'stalk-surface-above-ring', 'stalk-surface-below-ring',
              'stalk-color-above-ring', 'stalk-color-below-ring',
              'veil-type', 'veil-color', 'ring-number',
              'ring-type','spore-print-color',
              'population', 'habitat']

    df = pd.read_csv(data_file_name, sep=',', names=header)

    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    
    df['class'].replace('p', 0, inplace=True)
    df['class'].replace('e', 1, inplace=True)

    cols_to_transform = header[0:]
    df = pd.get_dummies(df, columns=cols_to_transform)

    df_train, df_test = train_test_split(df, test_size=0.9)

    num_train_entries = df_train.shape[0]
    num_train_features = df_train.shape[1] - 1

    num_test_entries = df_test.shape[0]
    num_test_features = df_test.shape[1] - 1

    df_train.to_csv('train_temp.csv', index=False)
    df_test.to_csv('test_temp.csv', index=False)

    open("mushroom_train.csv", "w").write(str(num_train_entries) +
                                          "," + str(num_train_features) +
                                          "," + open("train_temp.csv").read())

    open("mushroom_test.csv", "w").write(str(num_test_entries) +
                                         "," + str(num_test_features) +
                                         "," + open("test_temp.csv").read())

    os.remove("train_temp.csv")
    os.remove("test_temp.csv")

#Run Main Data Prep
MUSHROOM_DATA_FILE = "mushroom.csv"
prepare_data(MUSHROOM_DATA_FILE)
