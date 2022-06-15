import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression

x,y = make_regression(n_samples=150,n_features=1,noise=30,random_state=5)

#plt.scatter(x,y)
#plt.show()

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

legendas=["Treino","Teste"]
quantidade=20
f,ax=plt.subplots(2,2,figsize=[12,12])
plt.setp(ax,xticks=np.arange(0,20,step=1))

rand=1
quantidade_k,res_treino,res_teste=retorna_resultado(rand,quantidade,dados,respostas)
ax[0,0].plot(res_treino)
ax[0,0].plot(res_teste)
ax[0,0].grid(True)
ax[0,0].set_title("Rand {}".format(rand))
ax[0,0].legend(legendas)

rand=5
quantidade_k,res_treino,res_teste=retorna_resultado(rand,quantidade,dados,respostas)
ax[0,1].plot(res_treino)
ax[0,1].plot(res_teste)
ax[0,1].grid(True)
ax[0,1].set_title("Rand {}".format(rand))
ax[0,1].legend(legendas)

rand=10
quantidade_k,res_treino,res_teste=retorna_resultado(rand,quantidade,dados,respostas)
ax[1,0].plot(res_treino)
ax[1,0].plot(res_teste)
ax[1,0].grid(True)
ax[1,0].set_title("Rand {}".format(rand))
ax[1,0].legend(legendas)

rand=15
quantidade_k,res_treino,res_teste=retorna_resultado(rand,quantidade,dados,respostas)
ax[1,1].plot(res_treino)
ax[1,1].plot(res_teste)
ax[1,1].grid(True)
ax[1,1].set_title("Rand {}".format(rand))
ax[1,1].legend(legendas)

plt.show()