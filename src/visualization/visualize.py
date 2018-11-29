# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:29:18 2018

@author: pillis.krisztian
"""
import numpy as np
import holoviews as hv
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot

hv.extension('bokeh')

PATH_TO_PREDICTORS = '../data/raw/predictors15.csv'
PATH_TO_TRAIN = '../data/raw/train15.csv'

def plot_boxwhiskers(df):
    title = "Áramtermelés eloszlása zónánként"
    boxwhisker = hv.BoxWhisker(df, ('ZONEID','Zóna'), ('POWER','Teljesítmény'), label=title)
    boxwhisker.options(show_legend=False, width=500,height=500)
    title2 = "Áramtermelés eloszlása óránként csoportosítva"
    boxwhisker2 = hv.BoxWhisker(df, ('HOUR','Óra'), ('POWER','Teljesítmény'), label=title2)
    boxwhisker2.options(show_legend=False, width=500,height=500)
    options = {'BoxWhisker': dict(width=500, height=500)}
    return (boxwhisker + boxwhisker2).options(options)


def plot_boxplots(df,plot_columns=4):
    number_of_columns = len(df.columns.values)
    whole_rows = int(number_of_columns / plot_columns)
    extrarow = 1 if number_of_columns % plot_columns > 0 else 0

    row = -1
    column = 0
    for i, col in enumerate(df.columns.values):

        last_number_of_columns = number_of_columns%plot_columns
        is_last_row = int(number_of_columns/plot_columns)*plot_columns <= i

        if(i%plot_columns == 0):
            row += 1
            column = 0
            if is_last_row:
                fig, ax = plt.subplots(1,last_number_of_columns,figsize=(20,5))
            else:
                fig, ax = plt.subplots(1,plot_columns,figsize=(20,5))
        else:
            column += 1
        df[[col]].plot(ax = ax[column], kind='box')
        
def plot_scatters(df, target_column):
    number_of_columns = len(df.columns.values)
    plot_columns = 5
    whole_rows = int(number_of_columns / plot_columns)
    extrarow = 1 if number_of_columns % plot_columns > 0 else 0

    row = -1
    column = 0
    for i, col in enumerate(df.columns.values):

        last_number_of_columns = number_of_columns%plot_columns
        is_last_row = int(number_of_columns/plot_columns)*plot_columns <= i

        if(i%plot_columns == 0):
            row += 1
            column = 0
            if is_last_row:
                fig, ax = plt.subplots(1,last_number_of_columns,figsize=(20,4))
            else:
                fig, ax = plt.subplots(1,plot_columns,figsize=(20,4))
        else:
            column += 1
        df.plot(x=[col], y=[target_column], kind="scatter", ax = ax[column] )
        
def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(10,8))         # Sample figsize in inches

    f = (
        df.loc[:, df.columns]    
    ).corr()
    
    sns.heatmap(f, annot=True, ax=ax, linewidths=.5, cmap="PuBu")

def plot_lags_and_auto(df, size=250):
    columns = len(df.columns)
    for col in df.columns:
        fig, ax = plt.subplots(1,2,figsize=(20,5))
        fig.suptitle(col, fontsize=16)
        lag_plot(df[col].tail(size),ax =ax[0])
        autocorrelation_plot(df[col].tail(size), ax = ax[1])
        
def plot_moving_average_heatmap(df):
    df_roll = df.copy()
    windows = np.arange(1,7)
    step = 24
    rolling_columns = df.columns
    for column in rolling_columns:
        for window in windows:    
            rolling_column = df_roll[column].rolling(window = window)
            df_roll[column+" mozgó átlaga, ablak:"+str(window)] = rolling_column.mean().shift(step)
        chunk = np.append(df_roll.drop(rolling_columns,axis=1).columns.values,'POWER')
        fig, ax = plt.subplots(figsize=(10,8))         # Sample figsize in inches
        f = (
            df_roll.loc[:, chunk]    
        ).corr()
        sns.heatmap(f, annot=True, ax=ax, linewidths=.5, cmap="PuBu", center=0.3)