import MetaTrader5 as mt5
from datetime import datetime
import pandas

mt5.initialize("login","server","password")

info = mt5.terminal_info()

if info.community_connection:
    
    ativo='WINJ22'

    info = mt5.symbol_info(ativo)

    info = info._asdict()

    for data in info:
        print(" {} = {} ".format(data, info[data]))

    mt5.shutdown()
