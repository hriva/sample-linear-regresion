#!/usr/bin/env python3

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Ingest
DATA = "src/assets/BTC-USD.csv"
DATA1 = "src/assets/ETH-USD.csv"
df1 = pd.read_csv(DATA, index_col="Date", parse_dates=True).sort_index(ascending=True)
df2 = pd.read_csv(DATA1, index_col="Date", parse_dates=True).sort_index(ascending=True)

## Get necesary predictive values only.
# Use double brackets to avoid sending pandas.core.series.Series instead of DataFrame
dfx = df2[["Adj Close"]]
dfy = df1[["Adj Close"]]
df1["Adj Close"].corr(df2["Adj Close"])

## Split
train_size = 0.8
if dfx.shape[0] != dfy.shape[0]:
    print("Sample Sizes ERROR")
    exit
x_train_size = round(dfx.shape[0] * train_size)  # We only need the rows.
x_test_size = x_train_size
y_train_size = round(dfx.shape[0] * train_size)  # We only need the rows.
y_test_size = y_train_size

x_train, x_test = dfx.iloc[:x_train_size], dfx.iloc[x_test_size:]
y_train, y_test = dfy.iloc[:y_train_size], dfy.iloc[y_test_size:]


# Regression
regressor = LinearRegression()
regressor.fit(X=x_train, y=y_train)
y_pred = regressor.predict(x_test)
print(y_pred)


# Training Sets
plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Bitcoin vs Ethereum")
plt.xlabel("Ethereum Closing Price")
plt.ylabel("Bitcoin Closing Price")
plt.show()


# Test set
plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Bitcoin vs Ethereum")
plt.xlabel("Ethereum Closing Price")
plt.ylabel("Bitcoin Closing Price")
plt.show()
