# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:29:18 2018

@author: pillis.krisztian
"""
import pandas as pd

PATH_TO_PREDICTORS = '../data/raw/predictors15.csv'
PATH_TO_TRAIN = '../data/raw/train15.csv'

def read():
    df_power = pd.read_csv(PATH_TO_TRAIN)
    df_features = pd.read_csv(PATH_TO_PREDICTORS)
    return (df_power,df_features)