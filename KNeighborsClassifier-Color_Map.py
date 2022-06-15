from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(4,4))
plt.subplot(1,1,1)

x1,y1 = make_classification(n_samples=150,n_features=2,n_redundant=0,n_informative=2,n_clusters_per_class=1,random_state=3,class_sep=2,n_classes=3)

plt.scatter(x1[:,0],x1[:,1],marker="o",c=y1,s=25)

knn = KNeighborsClassifier(n_neighbors= 5)

x_train, x_test, y_train, y_test = train_test_split(x1,y1,random_state= 1)

knn.fit(x_train, y_train)

print("Treino: {}\n".format(knn.score(x_train,y_train)))
print("Teste: {}\n".format(knn.score(x_test,y_test)))

x_min,x_max = x1[:,0].min() -1, x1[:,0].max()+1
y_min,y_max = x1[:,0].min() -1, x1[:,0].max()+1

xx, yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))

cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])

z=knn.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx,yy,z,cmap=cmap_light)
plt.scatter(x1[:,0],x1[:,1],c=y1,cmap=cmap_bold,edgecolor='gray',s=20)

plt.show()
