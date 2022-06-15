import MetaTrader5 as mt5
from datetime import datetime
import pandas

mt5.initialize("login","server","password")

info = mt5.terminal_info()

if info.community_connection:

    info = info._asdict()

    for k in info.keys():
        print(k," - > ", info[k])

    mt5.shutdown()
