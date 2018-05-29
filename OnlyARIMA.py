INPUT_FILE = "Ripple.csv"
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima_model import ARIMA
import math
color = sns.color_palette()
p,d,q = 1,1,0
split_ratio = 0.7
def predict_and_plot_closing_price():
    df = pd.read_csv(INPUT_FILE, parse_dates=['Date'], usecols=["Date", "Close"])
    df = df[::-1]
    df.columns = ["ds", "y"]
    print(df.head())
    data = np.array(df["y"])
    split_per = split_ratio*len(df)
    split_per = int(math.ceil(split_per))
    training, test = data[:split_per], data[split_per:]
    history = [x for x in training]
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat,obs))
predict_and_plot_closing_price()