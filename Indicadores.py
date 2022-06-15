from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import datetime

data = yf.download('PETR4.SA','2016-01-01',datetime.datetime.today())

periodo_sma=200
periodo_ema=50
periodo_rsi=14

def SMA(data,periodo_sma):
    
    data['SMA'] = data['Close'].rolling(periodo_sma).mean()

    return data['SMA']

def EMA(data,periodo_ema):

    data['EMA'] = data['Close'].ewm(span=periodo_ema).mean()

    return data['EMA']

def RSI(data,periodo_RSI):

    data['change'] = data['Close']-data['Close'].shift(1)

    data['gain'] = data.loc[data['change']>0,'change'].apply(abs)
    data.loc[(data['gain'].isna()),'gain'] = 0
    data.loc[0,'gain'] = np.NaN

    data['loss'] = data.loc[data['change']<0,'change'].apply(abs)
    data.loc[(data['loss'].isna()),'loss'] = 0
    data.loc[0,'loss'] = np.NaN

    periodo_RSI=14

    data['avg_gain'] = data['gain'].rolling(periodo_RSI).mean()
    data['avg_loss'] = data['loss'].rolling(periodo_RSI).mean()

    data['Data']=data.index

    data=data.reset_index(drop=True)
  
    first = data['avg_gain'].first_valid_index()
    
    for index,row in data.iterrows():

        if index == first:
            prev_avg_gain=row['avg_gain']
            prev_avg_loss=row['avg_loss']
        elif index > first:
            data.loc[index,'avg_gain'] = ((prev_avg_gain*(periodo_RSI-1))+row['gain'])/periodo_RSI
            prev_avg_gain = data.loc[index,'avg_gain']
            data.loc[index,'avg_loss'] = ((prev_avg_loss*(periodo_RSI-1))+row['loss'])/periodo_RSI
            prev_avg_loss = data.loc[index,'avg_loss']

    data['RS'] = data['avg_gain']/data['avg_loss']

    data['RSI'] = 100 - (100/(1+ data['RS']))

    return data['RSI']

def ICHIMOKU(data):
    high9=data['High'].rolling(9).max()
    low9=data['Low'].rolling(9).min()
    high26=data['High'].rolling(26).max()
    low26=data['Low'].rolling(26).min()
    high52=data['High'].rolling(52).max()
    low52=data['Low'].rolling(52).min()

    data['tenkansen']=(high9+low9)/2
    data['kijunsen']=(high26+low26)/2
    data['senkouA']=((data['tenkansen']+data['kijunsen'])/2).shift(26)
    data['senkouB']=((high52+low52)/2).shift(26)
    data['chikou']=data['Close'].shift(-26)
    data=data.iloc[26:]

    columns=['tenkansen','kijunsen','senkouA','senkouB','chikou']

    return data[columns]

def MACD(data):
    data['EMA_FAST'] = data['Close'].ewm(12).mean()
    data['EMA_SLOW'] = data['Close'].ewm(26).mean()
    data['MACD'] = data['EMA_FAST'] - data['EMA_SLOW']
    data['SINAL_MACD'] = data['MACD'].ewm(9).mean()

    columns=['MACD','SINAL_MACD']

    return data[columns]

data['Data']=data.index.values

dias=data['Data']

data=data.reset_index(drop=True)

data['RSI']=RSI(data,periodo_rsi)
data['SMA']=SMA(data,periodo_sma)
data['EMA']=EMA(data,periodo_ema)
data.append(ICHIMOKU(data), ignore_index=False, verify_integrity=False, sort=False)
data.append(MACD(data), ignore_index=False, verify_integrity=False, sort=False)

data['Data']=dias.values

data.set_index(data['Data'],inplace=True)

data.drop(data['Data'])

print(data)

fig,ax=plt.subplots()

plt.subplot(4,1,1)

plt.plot(data.index,data['Close'],label='Fechamento',alpha=0.5,lw=0.8)

plt.plot(data.index,data['SMA'],label='SMA',color='orange',lw=0.5)

plt.plot(data.index,data['EMA'],label='EMA',color='purple',lw=0.5)

plt.subplot(4,1,2)

plt.plot(data.index,data['tenkansen'],lw=0.5)
plt.plot(data.index,data['kijunsen'],lw=0.5)
plt.plot(data.index,data['chikou'],lw=0.5)

komu = data['Adj Close'].plot(lw=0, color='b')
komu.fill_between(data.index,data['senkouA'],data['senkouB'], where=data['senkouA']>=data['senkouB'], color='lightgreen')
komu.fill_between(data.index,data['senkouA'],data['senkouB'], where=data['senkouA']<data['senkouB'], color='lightcoral')

plt.subplot(4,1,3)

plt.plot(data.index,data['RSI'],label='RSI',lw=0.8)

plt.axhline(y = 70, color = 'g', linestyle = 'dashed',lw=0.8)

plt.axhline(y = 30, color = 'g', linestyle = 'dashed',lw=0.8)

plt.subplot(4,1,4)

plt.plot(data.index,data['MACD'],color='orange',lw=0.5)

plt.plot(data.index,data['SINAL_MACD'],color='purple',lw=0.5)

data['HISTOGRAMA']=data['MACD']-data['SINAL_MACD']

positivo=data['HISTOGRAMA']>=0
negativo=data['HISTOGRAMA']<0

plt.bar(data.index[positivo],data['HISTOGRAMA'][positivo],color='green',width=0.8,align='center')
plt.bar(data.index[negativo],data['HISTOGRAMA'][negativo],color='red',width=0.8,align='center')

plt.show() 

#ax.plot()

#ax.plot(output_df['Data'],output_df['SMA50'],label='MMS50',color='brown')


#data['Anterior'] = data['SMA15'].shift(1) - data['SMA50'].shift(1)
#data['Atual'] =  data['SMA15'] - data['SMA50']

#data.loc[(data['Anterior']<=0)&(data['Atual']>0),'Compra']= data['Close']
#data.loc[(data['Anterior']>=0)&(data['Atual']<0),'Venda']= data['Close']


#ax.scatter(output_df['Data'],output_df['Compra'],label='Compra',marker='^',color='green')

#ax.scatter(output_df['Data'],output_df['Venda'],label='Venda',marker='v',color='red')
