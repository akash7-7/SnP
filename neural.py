#neural Net with Functions

#Neural Net for Stock Price Prediction

import os
os.chdir("C:\\Users\\509861\\Desktop\\SnP\\yahoo finance")
import pandas as pd
import numpy as np
import neurolab as nl

apple = pd.read_csv("apple.csv")
microsoft = pd.read_csv("microsoft.csv")
ibm = pd.read_csv("ibm.csv")

def returns(df, name):
    df.columns.values[-1] = 'AdjClose'
    df.columns = df.columns + '_' + name
    df['Return_%s' %name] = df['AdjClose_%s' %name].pct_change()
    return df

#stocks
apple = returns(apple,'APPL')
microsoft = returns(microsoft,'MSFT')
ibm = returns(ibm,'IBM')

def addFeatures(dataframe, adjclose, returns, n):
    return_n = adjclose[9:] + "Time" + str(n)
    dataframe[return_n] = dataframe[adjclose].pct_change(n)
    
    roll_n = returns[7:] + "RolMean" + str(n)
    dataframe[roll_n] = pd.rolling_mean(dataframe[returns], n)
    
def applyRollMeanDelayedReturns(datasets, delta):
    for dataset in datasets:
        columns = dataset.columns    
        adjclose = columns[-2]
        returns = columns[-1]
        for n in delta:
            addFeatures(dataset, adjclose, returns, n)
    
    return datasets  

#datasets
datasets = [apple,microsoft,ibm]
delta= range(len(datasets))

data = applyRollMeanDelayedReturns(datasets,delta)

apple= data[0]
microsoft = data[1]
ibm = data[2]

#modelling neural network

size = len(apple)

X = apple['Open_APPL']
y = apple['Return_APPL']
