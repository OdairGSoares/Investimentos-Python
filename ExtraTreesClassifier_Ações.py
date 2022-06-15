from statistics import mode
from matplotlib.pyplot import axis
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

df=yf.Ticker("PETR4.SA").history(period="max")

df=df.reset_index().drop('Date',axis=1).drop('Dividends',axis=1).drop('Stock Splits',axis=1)

amanha=df[-1::]

base=df.drop(df[-1::].index,axis=0)

base['target'] = base['Close'][1:len(base)].reset_index(drop=True)

prev=base[-1::].drop('target',axis=1)

treino=base.drop(base[-1::].index,axis=0)

treino.loc[treino['target']>treino['Close'],'target']=1
treino.loc[treino['target']!=1,'target']=0

y=treino['target']
x=treino.drop('target',axis=1)

x_treino,x_teste,y_treino,y_teste=train_test_split(x,y,test_size=0.3)

modelo=ExtraTreesClassifier()

modelo.fit(x_treino,y_treino)

resultado=modelo.score(x_teste,y_teste)

print("Win Rate: {}".format(resultado))

prev['target']=modelo.predict(prev)

print('-------------------------------')
print(prev.tail())
print(amanha)

