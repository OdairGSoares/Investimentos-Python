from fileinput import close
import MetaTrader5 as mt5
import pandas
import time

if mt5.initialize("login","server","password"):

    info = mt5.terminal_info()

    if info.community_connection:
        
        dias=100
        data=time.time()        
        carteira_acoes = ['ABEV3','ITUB4','ITSA4','VALE3','PETR4']
        carteira_closes = pandas.DataFrame()

        dataframe_closes=pandas.DataFrame(mt5.copy_rates_from('ITUB4',mt5.TIMEFRAME_D1,data,dias))

        for acao in carteira_acoes:

            closes=mt5.copy_rates_from(acao,mt5.TIMEFRAME_D1,data,dias)
            carteira_closes[acao] = closes['close']

        carteira_closes.set_index(dataframe_closes['time'], inplace= True)

        print(carteira_closes)

        mt5.shutdown()