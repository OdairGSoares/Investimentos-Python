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
import investpy as inv
import ta

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
    Profit_Retorno=(int(sum(take))/5)/Win
    Loss_Retorno=(((int(sum(lost))/5)/Loss)*100)/Profit_Retorno

    print('Win Rate: {}'.format(Win_Rate))
    print('Take: {} R$'.format(int(sum(take))/5))
    print('Loss: {} R$'.format(int(sum(lost))/5))
    print('Total: {} R$'.format((int(sum(take))-int(sum(lost)))/5))
    print('Risco_Retorno: {}'.format((100-Loss_Retorno)/Loss_Retorno))
     

    plt.figure(figsize=(16,16))
    plt.plot(grafico['Close'])
    plt.plot(grafico['Resultado'],color='orange',ls='--',lw=0.8)
    plt.show()

indices=inv.search_quotes(text='WIN',products=['indices'],countries=['brazil'],n_results=50)

for indice in indices[0:1]:
    print('=============================')

WIN=indice.retrieve_historical_data(from_date='01/01/1900',to_date='20/04/2024')

def preparar_info(WIN):

    WIN['Close']=WIN['Close'].shift(-1)

    WIN['mm']=WIN['Close'].rolling(8).mean()

    WIN=WIN.dropna()

    WIN=WIN.reset_index()

    WIN=WIN.drop('Date',axis=1)

    return WIN

WIN=preparar_info(WIN)

y=WIN['Close']

x=WIN.drop(['Close'],axis=1)

feature_list=('Open','High','Low','Volume','Change Pct','mm')

k_best_features=SelectKBest(k='all')

k_best_features.fit_transform(x,y)

k_best_features_scores=k_best_features.scores_

raw_pairs=zip(feature_list[1:],k_best_features_scores)

ordered_pairs=list(reversed(sorted(raw_pairs,key=lambda x: x[1])))

k_best_features_final=dict(ordered_pairs[:15])

best_features=k_best_features_final.keys()

print(k_best_features_final)

x=x.drop(['Open'],axis=1)

scaler=MinMaxScaler().fit(x)

feature_scale=scaler.transform(x)

x=pd.DataFrame(feature_scale)

x_treino,x_teste,y_treino,y_teste=train_test_split(x,y,test_size=0.3, shuffle=False)

#Regressão_Linear
modelo=linear_model.LinearRegression()

modelo.fit(x_treino,y_treino)

predicao=modelo.predict(x_teste)

coeficiente = r2_score(y_teste,predicao)

grafico=pd.DataFrame(y_teste).reset_index()

grafico['Resultado']=pd.DataFrame(predicao)[0]

grafico=grafico.drop('index',1) 

print(x_teste)

predict=modelo.predict(x_teste.iloc[-2:-1])

print(pd.DataFrame(predict)[0])

plot_graph(grafico,coeficiente)
