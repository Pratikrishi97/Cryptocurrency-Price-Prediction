INPUT_FILE = "Ripple3.csv"
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
import math
color = sns.color_palette()
p,d,q = 7,1,0
split_ratio = 0.7
def __getnewargs__(self):
	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
ARIMA.__getnewargs__ = __getnewargs__
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
        model_fit.save('model.pkl')
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat,obs))
    for t in range(7):
        loaded = ARIMAResults.load('model.pkl')
        output = loaded.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append(yhat)
        print('predicted=%f' % (yhat))
    plt.plot(test,color="#39FF14",linewidth=7.0)
    plt.plot(predictions, color="black")
    
predict_and_plot_closing_price()