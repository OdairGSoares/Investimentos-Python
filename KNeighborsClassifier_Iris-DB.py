from sklearn.datasets  import load_iris
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

dados_iris=load_iris()

x_train,x_test,y_train,y_test=train_test_split(dados_iris['data'],dados_iris['target'],random_state=0)

iris_dataframe=pd.DataFrame(x_train,columns=dados_iris["feature_names"])

grr=scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8)

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)

print("Treino: {}".format(knn.score(x_train,y_train)))
print("Teste: {}".format(knn.score(x_test,y_test)))
print(dados_iris.keys())
print(dados_iris["target_names"])
print("treino: {}".format(x_train.shape))
print("teste: {}".format(x_test.shape))

