import MetaTrader5 as mt5
from datetime import datetime
import pandas

mt5.initialize("login","server","password")

info = mt5.terminal_info()

if info.community_connection:
    
    ativos=mt5.symbols_get()

    for ativo in ativos:
        print(ativo.name)

    mt5.shutdown()
