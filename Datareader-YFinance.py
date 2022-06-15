import mplfinance as mpf
import pandas as pd
import yfinance as yf

df=yf.Ticker("PETR4.SA").history(period="max")

df = df.loc["2022-01-01":]

pd.DataFrame(df).to_csv("Dados_PETR4.csv", mode='a', header=False, index=False)

df=pd.read_csv("Dados_PETR4.csv",encoding='utf-8',index_col="Date",sep=',',usecols=['Open','High','Low','Close','Volume','Dividends','Stock Splits'])

mpf.plot(df, type = "candle",volume = True,style='yahoo')
