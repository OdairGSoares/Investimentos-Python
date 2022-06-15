from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(0,6,7).reshape(-1,1)
y=np.linspace(0,12,7).reshape(-1,1)

lr = LinearRegression().fit(x,y)

print("Coeficiente: {}".format(lr.coef_[0][0]))
print("Intercepto: {}".format(lr.intercept_[0]))
plt.plot(x,y)
plt.grid(True)
plt.show()