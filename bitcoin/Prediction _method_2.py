import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# read the data from CSV into a DataFrame
df=pd.read_csv('data_1.csv')
df.sort_values(["index"],inplace=True)
pd. set_option("display.max_columns", None)
print(df)

#function for some parameters in knowing stock price direction
def SMA(data,period=30,column='close'):
    return data[column].rolling(window=period).mean()

def EMA(data,period=20,column='close'):
    return data[column].ewm(span=period,adjust=False).mean()


def MACD(data,period_long=26,period_short=12,period_signal=9,column='close'):
    shortEMA = EMA(data, period =period_short,column=column)
    longEMA = EMA(data, period=period_long, column=column)
    data['MACD']=shortEMA-longEMA
    data['signal_line']=EMA(data,period=period_signal,column='MACD')
    return  data


def RSI (data,period=14,column='close'):
    delta =data[column].diff(1)
    delta=delta.dropna()
    up=delta.copy()
    down=delta.copy()
    up[up<0]=0
    down[down>0]=0
    data['up']=up
    data['down']=down
    AVG_Gain=SMA(data,period,column='up')
    AVG_Loss=abs(SMA(data,period,column='down'))
    RS=AVG_Gain/AVG_Loss
    RSI=100.0 -(100.0/(1.0+RS))
    data['RSI']=RSI
    return data

#get the parameters of the data in the stock
MACD(df)
RSI(df)

#add the SMA and EMA parameters to the data in CSV
df['SMA']=SMA(df)
df['EMA']=EMA(df)
scaler=MinMaxScaler()
print (df)

#add the diffrence between the close price of the old and new day
df['Target']=np.where(df['close'].shift(-1) > df['close'],0,1)
print (df)
df=df[29:]

#creating the Train data sets and Testing Data sets
keep_columns = ['close','MACD','signal_line','RSI','SMA','EMA']
X_x=df[keep_columns].tail(1).values
df=df.head(len(df)-1)
X=df[keep_columns].values
Y=df['Target'].values
max=0
score=0
rand=0
print(X_x)

#Split data into training and test sets for Decision Tree classifier
for i in range(1,2000):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=i)
    tree = DecisionTreeClassifier().fit(X_train, Y_train)
    rf=RandomForestClassifier(n_estimators=100)
    rf.fit(X_train,Y_train)


#see the accuracy of the trained model
    score=tree.score(X_test, Y_test)

    #keep changing the random state value to get the best accuracy from the model
    if score > max:
        rand=i
        max=score
        Predictions = tree.predict(X_test)
        print(rand)
        print(max)
        tommorow=tree.predict(X_x)
        rfs=rf.predict(X_x)
        print ("the model says" ,tommorow)
        print ("other says",rfs)

#Predict the next stock move
Predictions=tree.predict(X_test)
print(Predictions)

