# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:29:18 2018

@author: pillis.krisztian
"""
import pandas as pd

def read_data(train_path,predictor_path, join_features=True):
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
    return df.copy()
# -*- coding: utf-8 -*-
    
if __name__ == '__main__':
    read_data()