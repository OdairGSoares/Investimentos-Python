import MetaTrader5 as mt5
from datetime import datetime
import pandas

mt5.initialize("login","server","password")

info = mt5.terminal_info()

if info.community_connection:

    ativo = "PETR4"
    data = datetime(2021,1,2)
    ticks_num = 10
    
    dados = mt5.copy_ticks_from(ativo, data, ticks_num, mt5.COPY_TICKS_ALL)

    data = pandas.DataFrame(dados)

    print (data)

    mt5.shutdown()

    