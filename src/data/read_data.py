# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:29:18 2018

@author: pillis.krisztian
"""
import pandas as pd
import pickle
import os

dirname = os.path.dirname(__file__)

def read_data(train_path="../../data/raw/train15.csv",predictor_path="../../data/raw/predictors15.csv", join_features=True):
    DATE_FORMAT = '%Y%m%d %H:%M'
    INDEX_COLUMN = 'TIMESTAMP'
    dateparse = lambda x: pd.datetime.strptime(x, DATE_FORMAT)
    df = pd.read_csv(train_path)
    df[INDEX_COLUMN] = df[INDEX_COLUMN].apply(dateparse)
    if join_features:
        df_features = pd.read_csv(predictor_path)
        df_features[INDEX_COLUMN] = df_features[INDEX_COLUMN].apply(dateparse)
        df = df.merge(df_features, how='left', on=[INDEX_COLUMN,'ZONEID'])
    df.set_index(INDEX_COLUMN)
    pickle.dump(df, open("../../data/interim/data.p","wb"))
    return df.copy()
# -*- coding: utf-8 -*-
    
if __name__ == '__main__':
    read_data()