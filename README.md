# Description 
Small Example of a simple linear regression script. 

> DISCLAIMER: This script is for demonstrative propouses only, it should not be used in production without the proper analysis of the variables to predict, including but not limited to their autocorrelation/multicollinearity, correlation and or seasonality.

# Install
> This project requires   
 ![Python](https://img.shields.io/badge/Python-3.11.7-Blue?labelColor=White&style=flat). It is recommended to create a virtual enviorment to use it.

```sh
git clone --depth 1 https://github.com/hriva/sample-linear-regression.git 
cd sample-linear-regression

# Create virt env
virtualenv -p python3 sample-linear-regression
source sample-linear-regression/bin/activate
cd sample-linear-regression
pip3 install -r requirements.txt

# Run
chmod +x src/bitcoin-ethereum.py
./src/bitcoin-ethereum.py
```
# Explanation

# Linear Regression

Small linear regression sample implementation.
For this example we are using the Bitcoin Prices in a Monthly basis as the **dependant** variable. i.e., the variable we want to predict. 
And we use the Ethereum Prices in a Montly basis as the **independant** variable. i.e., the variable we are using **to** predict the Bitcoin price. 

> DISCLAIMER: this is simplified given that yahoo offers good quality data. Usually you need more steps to clean and wrangle data.

## 1. Import libraries


```python
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
```

## 2. Load the data from csv files.


```python
# Ingest
## Data paths.
DATA = "src/assets/BTC-USD.csv" 
DATA1 = "src/assets/ETH-USD.csv"

## Make pandas read the data.
df1 = pd.read_csv(DATA, index_col="Date", parse_dates=True).sort_index(ascending=True)
df2 = pd.read_csv(DATA1, index_col="Date", parse_dates=True).sort_index(ascending=True)
```

Ingests the data fetched from Yahoo Finance (The data has no blank values).
During the import, the data is formated to a time series by setting the Dates as the index. The index is then sorted as ascend given that for linear regresions these need to be from older to newest.


```python
df1.head()
df2.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-02-01</th>
      <td>107.147682</td>
      <td>165.549622</td>
      <td>102.934563</td>
      <td>136.746246</td>
      <td>136.746246</td>
      <td>101430995445</td>
    </tr>
    <tr>
      <th>2019-03-01</th>
      <td>136.836243</td>
      <td>149.613235</td>
      <td>125.402702</td>
      <td>141.514099</td>
      <td>141.514099</td>
      <td>138882123600</td>
    </tr>
    <tr>
      <th>2019-04-01</th>
      <td>141.465485</td>
      <td>184.377853</td>
      <td>140.737564</td>
      <td>162.166031</td>
      <td>162.166031</td>
      <td>204556824026</td>
    </tr>
    <tr>
      <th>2019-05-01</th>
      <td>162.186554</td>
      <td>287.201630</td>
      <td>159.660217</td>
      <td>268.113556</td>
      <td>268.113556</td>
      <td>314349041886</td>
    </tr>
    <tr>
      <th>2019-06-01</th>
      <td>268.433350</td>
      <td>361.398682</td>
      <td>229.257431</td>
      <td>290.695984</td>
      <td>290.695984</td>
      <td>270589672710</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Preprocess the data for the regression.


```python
## Get the correlation
df1["Adj Close"].corr(df2["Adj Close"])
```




    0.9187847614681434




```python
## Get necesary predictive values only.
# Use double brackets to avoid sending pandas.core.series.Series instead of DataFrame
dfx = df2[["Adj Close"]]  # Load the Etherum price as x (Independant)
dfy = df1[["Adj Close"]]  # Load the Bitcoin price as y (Dependant)
```

Viewing the data shows the varying measures for the exchange prices. We need to **drop** all of them except the "Adj Close".

Notice that unlike the correlation. We fetch the columns using **double brackets**. 
This is to avoid getting errors in scikit-learn given that we are using 1 variable as preditor instead of a multy plexed array (a matrix).

## 4. Split the sets


```python
train_size = 0.8  # use 80 percent to train the regression
if dfx.shape[0] != dfy.shape[0]:
    print("Sample Sizes ERROR")
    exit
x_train_size = round(dfx.shape[0] * train_size)  # We only need the rows.
x_test_size = x_train_size
y_train_size = round(dfx.shape[0] * train_size)  # We only need the rows.
y_test_size = y_train_size

x_train, x_test = dfx.iloc[:x_train_size], dfx.iloc[x_test_size:]
y_train, y_test = dfy.iloc[:y_train_size], dfy.iloc[y_test_size:]
```

Why not use all the series for the regression?
To avoid overfitting. 

## 5. Regression


```python
# Regression
regressor = LinearRegression()
regressor.fit(X=x_train, y=y_train)
```

Create a Linear Regression instance and then fit it to the linear regression we need.  
**x** is the Ethereum price.  
**y** is the Bitcoin price.  

## 6. Predict 


```python
y_pred = regressor.predict(x_test)
print(y_pred)
```

    [[29754.22137026]
     [30433.23886914]
     [30398.68553649]
     [31129.1065507 ]
     [30176.4599037 ]
     [27572.75954253]
     [27888.41823535]
     [29685.41814795]
     [32605.41128462]
     [35436.57757274]
     [35799.63595354]
     [34648.96926019]]


Create an array with the predictions for the **test (Validation)** set.  
Print the predictions. 

## 7. Plot.


```python
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

```


    
![png](bitcoin-ethereum_files/bitcoin-ethereum_22_0.png)
    



    
![png](bitcoin-ethereum_files/bitcoin-ethereum_22_1.png)
