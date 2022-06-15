#import das bibliotecasa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

#ler base de dados
arquivo = pd.read_csv('C:/Users/odago/Desktop/mql5-python-projects/data/wine_dataset.csv')

#transformar os dados de predição em numeros para que o algoritmo reconheça
arquivo['style'] = arquivo['style'].replace('red',0)
arquivo['style'] = arquivo['style'].replace('white',1)

#separar os dados de avaliação e de predição
y=arquivo['style']
x=arquivo.drop('style',axis=1)

#separar dados de treino e teste
x_treino,x_teste,y_treino,y_teste=train_test_split(x,y,test_size=0.3)

#definir modelo
modelo=ExtraTreesClassifier(n_estimators=100)

#treino
modelo.fit(x_treino,y_treino)

#teste
resultado=modelo.score(x_teste,y_teste)

#printar resultado do teste
print("Win Rate: {}".format(resultado))

previsoes=modelo.predict(x_teste[400:403])

print('respostas: {}'.format(y_teste[400:403]))
print('previsoes: {}'.format(previsoes))


