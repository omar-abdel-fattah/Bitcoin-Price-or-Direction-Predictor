from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#import data from csv to Data frame
df=pd.read_csv('olymp.csv')

#sort the data
df.sort_values(["date"],inplace=True)
print(df)
F=21
actual_price =df.tail(1)
print (actual_price)
df=df.head(len(df)-1)
df=df.tail(F)
print(df)

days=list()
close=list()

df_days=df.loc[:,'date']
df_close=df.loc[:,'close']

X=list(df_days.index)
for x in range (0,len(X)):
    days.append([x])
print(days)

for closep in df_close:
    close.append(float(closep))
print(close)

#use SVR method to train the data set
rbf_svr=SVR(kernel='rbf',C=1000.0,gamma=0.85)
rbf_svr.fit(days,close)

#plot the real price of the stock
plt.figure(figsize=(16,8))
plt.scatter(days,close,color='black',label='Data')
days.append([F])
#days.append([22])
#days.append([23])

#perdict the price after x days and plot the real and the predicted proce to test the accuarcy of the model
plt.plot(days,rbf_svr.predict(days),color='blue',label='RBF')
plt.legend()
plt.show()

day=[[F]]

print('predicted price=',rbf_svr.predict(day))

