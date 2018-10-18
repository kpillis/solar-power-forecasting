# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:29:18 2018

@author: pillis.krisztian
"""

def prepare(df, predict_interval, rolling_intervals):
    
    INDEX_COLUMN = 'TIMESTAMP'

    
    
    
    df["ZONE_1"] = df["ZONEID"].apply(lambda x: x == 1)
    df["ZONE_2"] = df["ZONEID"].apply(lambda x: x == 2)
    df["ZONE_3"] = df["ZONEID"].apply(lambda x: x == 3)
    #df = df[df["ZONEID"] == 1]
    df = df.drop("ZONEID",axis=1)
    
    df["MONTH"] = df[INDEX_COLUMN].apply(lambda x: x.month)
    df["HOUR"] = df[INDEX_COLUMN].apply(lambda x: x.hour)
    df = df.drop("TIMESTAMP",axis=1)
    #df = df.set_index("TIMESTAMP")
    
    return df

def clear_data_from_end(df, columns, until):
    for column in columns:
        for i in range(until):    
            df.iloc[-1*(i+1),df.columns.get_loc(column)] = None
    return df

def add_rolling(df,columns, intervals, shift):
    for column in columns:
        for i in intervals:
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
    return df

def dissipate_features(df, columns):
    for column in columns:
        df[column] = df[column].rolling(window=2).apply(lambda x: x[1] if x[1] - x[0] < 0 else x[1] - x[0], raw='True')
    return df


if __name__ == '__main__':
    prepare()