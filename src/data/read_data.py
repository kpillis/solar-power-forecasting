# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:29:18 2018

@author: pillis.krisztian
"""
import pandas as pd

def read_data(train_path,predictor_path, join_features=True):
    
    INDEX_COLUMN = 'TIMESTAMP'
    df = pd.read_csv(train_path)
    if join_features:
        df_features = pd.read_csv(predictor_path)
        df = df.merge(df_features, how='left', on=[INDEX_COLUMN,'ZONEID'])
    return df.copy()
# -*- coding: utf-8 -*-
    
if __name__ == '__main__':
    read_data()