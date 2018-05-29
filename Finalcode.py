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
        #print('predicted=%f, expected=%f' % (yhat,obs))
    """for t in range(150):
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append(yhat)
        print('predicted=%f' % (yhat))"""
    #df=df[::-1]
    #print(df["y"].head())
    xy=df["y"].values
    #plt.plot(xy)
    p1=[]
    for i in range(len(training)):
        p1.append(training[i])
    for i in range(len(predictions)):
        p1.append(predictions[i])
    """plt.plot(xy,color="red")
    #print(p1)
    plt.plot(p1,color="blue")
    #print(predictions)
    """
    
    df1 = df['ds'].apply(lambda x: mdates.date2num(x))
    fig, ax = plt.subplots(figsize=(12, 8))
    #plot the mean predicted value
    sns.tsplot(p1, time=df1, alpha=0.8, color="red", ax=ax)
    #plot the actual value
    #plt.scatter(df, X,color="orange", alpha=0.3)
    #set the major axis as the date
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))
    #configure the plot
    fig.autofmt_xdate()
    #plot the upper and lower bounded regions and fill it
    plt.plot_date(df1,p1,'-',color="#66b3ff")
    plt.plot_date(df1,xy,'-',color="red")
    #plt.plot_date(future, df[columns[2]],'-',color="#004080")
    #plt.fill_between(future, df[columns[1]], df[columns[2]],facecolor='blue', alpha=0.1, interpolate=True)
    # Place a legend to the right of this smaller subplot.
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price in USD', fontsize=12)
    plt.title(title, fontsize=15)
    plt.show()
    
    
    
    
    
    
    
    
predict_and_plot_closing_price()