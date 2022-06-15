from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import datetime
import ta

data = yf.download('PETR4.SA','2016-01-01',datetime.datetime.today())

bb_indicatorl= ta.volatility.bollinger_lband(data['Close'],14,2,False)
bb_indicatorh= ta.volatility.bollinger_hband(data['Close'],14,2,False)
bb_indicatorm= ta.volatility.bollinger_mavg(data['Close'],14,False)

plt.plot(data.index,data['Close'],label='Fechamento',alpha=0.5,lw=0.8)
plt.plot(data.index,bb_indicatorl,color='orange',lw=0.5)
plt.plot(data.index,bb_indicatorh,color='orange',lw=0.5)
plt.plot(data.index,bb_indicatorm,color='orange',lw=0.5)

plt.show() 
