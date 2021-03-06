import quandl
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = "7HCx_y8DkJyhCMtUcvX5"

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / \
    df['Adj. Open'] * 100.0

#Final Dataframe and features
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#Assigns Label Column (what you're trying to predict)
forecast_col = 'Adj. Close'
#This fills in missing data
df.fillna(-9999, inplace=True)
#Used to predict out the data 1% of the current data
forecast_out = int(math.ceil(0.01*len(df)))

#Essentially shifts the columns negeatively --> spreadsheet shifted up so each labeled column is the adjusted close price 1% of days into future
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

print(df.head())
