import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression

x,y = make_regression(n_samples=150,n_features=1,noise=30,random_state=5)

def retorna_resultado(random_state,quantidade,dados,respostas):
    x_train,x_test,y_train,y_test=train_test_split(dados,respostas,random_state=random_state)
    
    quantidade_k=range(1,quantidade+1)
    res_teste=[]
    res_treino=[]

    for i in quantidade_k:
        knn=KNeighborsRegressor(n_neighbors=i)
        knn.fit(x_train,y_train)

        res_treino.append(knn.score(x_train,y_train))
        res_teste.append(knn.score(x_test,y_test))

    return quantidade_k, res_treino, res_teste

dados=x
respostas=y

legendas=["Linha de predição","Treino","Teste"]
f,ax=plt.subplots(1,3,figsize=[15,5])
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)

linha = np.linspace(-3,3,1000).reshape(-1,1)

for n_neighbors, ex in zip([5,7,9], ax):
    reg=KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(x_train,y_train)
    ex.plot(linha,reg.predict(linha))
    ex.plot(x_train,y_train,'^',markersize=5)
    ex.plot(x_train,y_train,'v',markersize=5)

    ex.set_title("{} neighbors\n Treino: {:.2f} - Teste: {:.2f}".format(n_neighbors,reg.score(x_train,y_train),reg.score(x_test,y_test)))

ax[0].legend(legendas)

plt.plot()

plt.show()