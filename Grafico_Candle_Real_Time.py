from time import sleep, time
from unicodedata import numeric
import MetaTrader5 as mt5
import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas
import csv
from matplotlib import animation
from datetime import datetime

if mt5.initialize("login","server","password") and mt5.terminal_info().community_connection:

        ativo='WINJ22' 
        periodo=mt5.TIMEFRAME_M1
        hoje=datetime.now()
        ticks=1  
        dados_desde="2022-01-01"

        if mt5.symbol_select(ativo,True):

            def animar(i):

                rates=mt5.copy_rates_from(ativo,periodo,hoje,ticks)     

                dataframe_rates=pandas.DataFrame(rates)

                dataframe_rates.drop(['spread','tick_volume','real_volume'],axis=1,inplace=True)

                dataframe_rates=pandas.DataFrame(dataframe_rates.iloc[-1]).T

                dataframe_rates.rename(columns = {'time':'Date', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close'}, inplace = True)

                dataframe_rates['Date']=pandas.to_datetime(dataframe_rates['Date'], unit='s')

                info=pandas.read_csv("Dados.csv")

                if str(info.iloc[-1]['Close'])!=str(dataframe_rates['Close'][0]) or str(info.iloc[-1]['Open'])!=str(dataframe_rates['Open'][0]) or str(info.iloc[-1]['High'])!=str(dataframe_rates['High'][0]) or str(info.iloc[-1]['Low'])!=str(dataframe_rates['Low'][0]) or str(info.iloc[-1]['Date'])!=str(dataframe_rates['Date'][0]):
                    pandas.DataFrame(dataframe_rates).to_csv("Dados.csv", mode='a', header=False, index=False)

                info=pandas.read_csv("Dados.csv")

                lista=[]

                for i in info.index:
                    if i>0:                     
                        if info.loc[i]['Date']!=info.loc[i-1]['Date']:
                            lista.append([info.loc[i-1]['Date'],info.loc[i-1]['Open'],info.loc[i-1]['High'],info.loc[i-1]['Low'],info.loc[i-1]['Close']])
                    elif i==0:
                        lista.append([dataframe_rates['Date'][0],dataframe_rates['Open'][0],dataframe_rates['High'][0],dataframe_rates['Low'][0],dataframe_rates['Close'][0]])

                candle_atual=lista[0]
            
                lista.pop(0)

                lista.append(candle_atual)

                ultimo_close_minuto=pandas.DataFrame(lista,columns=['Date','Open','High','Low','Close'])

                ultimo_close_minuto=ultimo_close_minuto.set_index(pandas.DatetimeIndex(ultimo_close_minuto['Date']))

                ultimo_close_minuto = ultimo_close_minuto.loc[dados_desde:]

                ax1.clear()

                mpf.plot(ultimo_close_minuto,ax=ax1, type = "candle",style='yahoo')
                
        fig=mpf.figure(style="yahoo",figsize=(7,8))

        ax1 = fig.add_subplot(1,1,1)
    
        ani=animation.FuncAnimation(fig,animar,interval=0)

        mpf.show()

        #mt5.shutdown()