#!/usr/bin/env python3.11

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Ingest
DATA =  "src/assets/BTC-USD.csv"
DATA1 = 'src/assets/ETH-USD.ETH-USD.csv'
dfy = pd.read_csv(DATA, index_col= 'Date', parse_dates = True)
dfx = pd.read_csv(DATA, index_col= 'Date', parse_dates = True)

## Get necesary predictive values only.
dfx = dfx["Adj Close"]
dfy = dfy["Adj Close"]

# Preprocess
# dfy['Date'] = pd.to_datetime(dfy['Date']) #  We did this during import
# dfy.set_index('Date',inplace=True)  #  We did this during import

## Split
x_train, x_test, y_train, y_test = train_test_split(
    dfx,dfy,test_size =.2, random_state =0) 

# Describe
dfy.describe()
dfx.describe()

# Regression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
#using fit works like a statement
    #FIT this OBJECT to THESE PARAMETERS 
#SOME METHODS TRANSFORM OBJECTS 
#SOME USE THEM AS INPUT
y_pred = regressor.predict(x_test)




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

