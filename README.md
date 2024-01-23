# Description 
Small Example of a simple linear regression script. 

> DISCLAIMER: This script is for demonstrative propouses only, it should not be used in production without the proper analysis of the variables to predict, including but not limited to their autocorrelation/multicollinearity, correlation and or seasonality.

# Install
> This project requires   
 ![Python](https://img.shields.io/badge/Python-3.11.7-Blue?labelColor=White&style=flat). It is recommended to create a virtual enviorment to use it.

```sh
git clone https://github.com/hriva/sample-linear-regression.git
cd sample-linear-regression

# Create virt env
python3.11 venv -m .venv
./venv/bin/activate
pip install -U pip setuptools wheel
pip install -rU requirements.txt
```