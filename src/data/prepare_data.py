# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:29:18 2018

@author: pillis.krisztian
"""
import pickle

ONE_DAY = 24
ONE_WEEK = 7 * ONE_DAY
ONE_MONTH = 30* ONE_DAY

def prepare(df=None, TRAIN_SIZE=ONE_MONTH, PREDICT_INTERVAL=ONE_WEEK):
    if df == None:
        df = pickle.load(open('../../data/interim/data.p','rb'))
    
    df = df[:TRAIN_SIZE]
    
    INDEX_COLUMN = 'TIMESTAMP'
    TARGET_COLUMN = 'POWER'
    feature_columns = ['VAR78', 'VAR79', 'VAR134', 'VAR157', 'VAR164',
       'VAR165', 'VAR166', 'VAR167','VAR169', 'VAR175', 'VAR178', 'VAR228']
    ACCUMLATED_FEATURE_COLUMNS = ['VAR169', 'VAR175', 'VAR178', 'VAR228']

    df["MONTH"] = df[INDEX_COLUMN].apply(lambda x: x.month)
    df["HOUR"] = df[INDEX_COLUMN].apply(lambda x: x.hour)
    
    df["ZONE_1"] = df["ZONEID"].apply(lambda x: x == 1)
    df["ZONE_2"] = df["ZONEID"].apply(lambda x: x == 2)
    df["ZONE_3"] = df["ZONEID"].apply(lambda x: x == 3)
    df = df.drop("ZONEID",axis=1)
    
    #TODO: ezt nem itt kéne
    df = clear_data_from_end(df,TARGET_COLUMN,PREDICT_INTERVAL)
    df = add_rolling(df,TARGET_COLUMN, [ONE_DAY, ONE_WEEK, ONE_MONTH], PREDICT_INTERVAL)

    for column in ACCUMLATED_FEATURE_COLUMNS:
        df[column] = dissipate_features(df,column)
        
    for column in feature_columns:
        #TODO: ezt nem itt kéne
        df = clear_data_from_end(df,column,PREDICT_INTERVAL)
        df = add_rolling(df,column, [ONE_DAY, ONE_WEEK], PREDICT_INTERVAL)
        df = df.drop(column, axis = 1)
    
    pickle.dump(df, open('../../data/processed/data.p', 'wb'))
    return df

def clear_data_from_end(df, column, until):
    for i in range(until):
        df.iloc[-1*(i+1),df.columns.get_loc(column)] = None
    return df

def add_rolling(df,column, intervals, shift):
    for i in range(min(intervals),max(intervals)):
    #for i in intervals:
        if i >= shift:
            rolling_column = df[column].rolling(window = i)
            df["ROLLING_MEAN_"+column+"_"+str(i)] = rolling_column.mean().shift(shift)
            df["ROLLING_MIN_"+column+"_"+str(i)] = rolling_column.min().shift(shift)
            df["ROLLING_MAX_"+column+"_"+str(i)] = rolling_column.max().shift(shift)
            df["ROLLING_SUM_"+column+"_"+str(i)] = rolling_column.sum().shift(shift)
            df["ROLLING_MEDIAN_"+column+"_"+str(i)] = rolling_column.median().shift(shift)
            df["ROLLING_STD_"+column+"_"+str(i)] = rolling_column.std().shift(shift)
            df["ROLLING_VAR_"+column+"_"+str(i)] = rolling_column.var().shift(shift)
            df["ROLLING_SKEW_"+column+"_"+str(i)] = rolling_column.skew().shift(shift)
            df["ROLLING_KURT_"+column+"_"+str(i)] = rolling_column.kurt().shift(shift)
            #df[column] =  df[column].shift(shift)
    return df

def dissipate_features(df, column):
    return df[column].rolling(window=2).apply(lambda x: x[1] if x[1] - x[0] < 0 else x[1] - x[0])


if __name__ == '__main__':
    prepare()