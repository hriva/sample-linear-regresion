#!/usr/bin/env python3.11

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Ingest
DATA =  "src/assets/BTC-USD.csv"
DATA1 = 'src/assets/ETH-USD.csv'
df1 = pd.read_csv(DATA)#, index_col= 'Date', parse_dates = True)
df2 = pd.read_csv(DATA1)#, index_col= 'Date', parse_dates = True)

## Get necesary predictive values only.
dfx = df2['Adj Close'].sort_index(ascending=False)
dfy = df1["Adj Close"].sort_index(ascending=False)
dfx.corr(dfy)

## Split
train_size = 0.8
if dfx.shape[0] != dfy.shape[0]:
    print("Sample Sizes ERROR")
    exit
x_train_size = round(dfx.shape[0] * train_size)  # We only need the rows.
x_test_size =  x_train_size 
y_train_size = round(dfx.shape[0] * train_size)  # We only need the rows.
y_test_size = y_train_size 

x_train, x_test = dfx.iloc[:x_train_size], dfx.iloc[x_test_size:]
y_train, y_test = dfy.iloc[:y_train_size], dfy.iloc[y_test_size:]


# Regression
regressor = LinearRegression()
regressor.fit(X=x_train,y=y_train)
#using fit works like a statement
    #FIT this OBJECT to THESE PARAMETERS 
#SOME METHODS TRANSFORM OBJECTS 
#SOME USE THEM AS INPUT
y_pred = regressor.predict(x_test)
LinearRegression.fit()



# Training Sets
plt.scatter(x_train,y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salaries in function of Exp ')
plt.xlabel('Experience in years')
plt.ylabel('Salary')
plt.show()


# Test set
plt.scatter(x_test,y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salaries in function of Exp ')
plt.xlabel('Experience in years')
plt.ylabel('Salary')
plt.show()

