from functools import total_ordering
from telnetlib import GA
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def plot_graph(grafico,coeficiente):

    print(f'Coeficiente de determinação: {coeficiente*100:.2f}')

    grafico['Close']=grafico['Close'].shift(1)

    Win=0
    Loss=0
    take=[]
    lost=[]

    for x in grafico.index:
        if grafico['Close'].iloc[x] > grafico['Close'].iloc[x-1] and grafico['Resultado'].iloc[x] > grafico['Resultado'].iloc[x-1] or grafico['Close'].iloc[x] < grafico['Close'].iloc[x-1] and grafico['Resultado'].iloc[x] < grafico['Resultado'].iloc[x-1]:
            Win=Win+1
            if grafico['Close'].iloc[x] > grafico['Close'].iloc[x-1]:
                take.append(grafico['Close'].iloc[x]-grafico['Close'].iloc[x-1])
            else:
                take.append(grafico['Close'].iloc[x-1]-grafico['Close'].iloc[x])
        else:
            if grafico['Close'].iloc[x] > grafico['Close'].iloc[x-1]:
                lost.append(grafico['Close'].iloc[x]-grafico['Close'].iloc[x-1])
            else:
                lost.append(grafico['Close'].iloc[x-1]-grafico['Close'].iloc[x])
            Loss=Loss+1

    lost = [x for x in lost if np.isnan(x) == False]
    take = [x for x in take if np.isnan(x) == False]

    Total=Win+Loss
    Win_Rate=(Win*100)/Total
    Total_Retorno=(int(sum(lost))+int(sum(take)))

    print('Win Rate: {}'.format(Win_Rate))
    print('Take: {}'.format(int(sum(take))))
    print('Loss: {}'.format(int(sum(lost))))
    print('Total: {}'.format((int(sum(take))-int(sum(lost)))))
    print('Risco_Retorno: {}'.format(((100*int(sum(take)))/Total_Retorno)/((100*int(sum(lost)))/Total_Retorno)))
     

    plt.figure(figsize=(16,16))
    plt.plot(grafico['Close'])
    plt.plot(grafico['Resultado'],color='orange',ls='--',lw=0.8)
    plt.show()

df=yf.Ticker("WINc1").history(period="max")

print(df)

df['mm5d']=df['Close'].rolling(5).mean()
df['mm21d']=df['Close'].rolling(21).mean()
df['Close']=df['Close'].shift(-1)
df=df.dropna()
df=df.reset_index()
df=df.drop('Date',axis=1)
df_prev=df.iloc[-1]

print(df_prev)

y=df['Close']
x=df.drop(['Close'],axis=1)

feature_list=('Open','High','Low','Volume','Dividends','Stock Splits','mm5d','mm21d')
k_best_features=SelectKBest(k='all')
k_best_features.fit_transform(x,y)
k_best_features_scores=k_best_features.scores_
raw_pairs=zip(feature_list[1:],k_best_features_scores)
ordered_pairs=list(reversed(sorted(raw_pairs,key=lambda x: x[1])))
k_best_features_final=dict(ordered_pairs[:15])
best_features=k_best_features_final.keys()

print("Melhores features: ")
print(k_best_features_final)

x=x.drop(['Open','Dividends','mm5d'],1)

scaler=MinMaxScaler().fit(x)
feature_scale=scaler.transform(x)
x=pd.DataFrame(feature_scale)

x_treino,x_teste,y_treino,y_teste=train_test_split(x,y,test_size=0.4, shuffle=False)

print('----------------------------')

#Rede_Neural
rn=MLPRegressor(max_iter=2000)
rn.fit(x_treino,y_treino)
pred=rn.predict(x_teste)
c=rn.score(x_teste,y_teste)

#Regressão_Linear
lr=linear_model.LinearRegression()
lr.fit(x_treino,y_treino)
pred=lr.predict(x_teste)
c = r2_score(y_teste,pred)

grafico=pd.DataFrame(y_teste).reset_index()
grafico['Resultado']=pd.DataFrame(pred)[0]
grafico=grafico.drop('index',1)

plot_graph(grafico,c)

def regressao_parametrizada():

    rn=MLPRegressor()

    parameter_space={
        'hidden_layer_sizes':[(i,) for i in list(range(1,21))],
        'activation':['tanh','relu'],
        'solver':['sgd','adam','lbfgs'],
        'alpha':[0.0001,0.05],
        'learning_rate':['constant','adaptive'],
    }

    search=GridSearchCV(rn,parameter_space,n_jobs=-1,cv=5)
    search.fit(x_treino,y_treino)
    clf=search.best_estimator_
    pred=search.predict(x_teste)
    cd=search.score(x_teste,y_teste)
