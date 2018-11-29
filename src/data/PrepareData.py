# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:29:18 2018

@author: pillis.krisztian
"""
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

INDEX_COLUMN = 'TIMESTAMP'
JOIN_COLUMNS = ['TIMESTAMP','ZONEID']
ACCUMLATED_COLUMNS = ['SOLAR_RAD','TERMAL_RAD','TOP_NET_SOLAR_RAD','TOTAL_PRECIPATION']
ZONE = 'ZONEID'
DATE_FORMAT = '%Y%m%d %H:%M'
dateparse = lambda x: pd.datetime.strptime(x, DATE_FORMAT)

def transform_outliers(df):
    median = df.loc[df['TOP_NET_SOLAR_RAD']<5000000, 'TOP_NET_SOLAR_RAD'].mean()
    df["TOP_NET_SOLAR_RAD"] = np.where(df["TOP_NET_SOLAR_RAD"] >5000000, median,df['TOP_NET_SOLAR_RAD'])
    median = df.loc[df['SOLAR_RAD']<3500000, 'SOLAR_RAD'].mean()
    df["SOLAR_RAD"] = np.where(df["SOLAR_RAD"] >5000000, median,df['SOLAR_RAD'])
    return df

def join(df_power,df_features):
    df_joined = df_power.merge(df_features, how='left', on=JOIN_COLUMNS)
    return df_joined

def split_by_zone(df):
    dfs = [df[df[ZONE] == 1],df[df[ZONE] == 2],df[df[ZONE] == 3]]
    return dfs
    
def dissipate_features(df, columns=ACCUMLATED_COLUMNS):
    for column in columns:
        df[column] = df[column].rolling(window=2).apply(lambda x: x[1] if x[1] - x[0] < 0 else x[1] - x[0], raw='True')
    return df

def rename_columns(df,column_mapping):
    return df.rename(index=str, columns=column_mapping)

def set_timestamp_as_index(df):
    df[INDEX_COLUMN] = df[INDEX_COLUMN].apply(dateparse)
    df = df.set_index(INDEX_COLUMN)
    return df
    
def extract_data_from_timestamp(df):
    df["YEAR"] = df.index.year
    df["MONTH"] = df.index.month
    df["DAY"] = df.index.day
    df["HOUR"] = df.index.hour
    return df