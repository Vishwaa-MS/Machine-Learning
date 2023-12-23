import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

iriset = datasets.load_iris()
X = iriset.data[:50,0:1] # only sepal length [50 rows, 1st column]
y = iriset.data[:50,1]  # only sepal width [50 rows, 2nd column]

reg = LinearRegression().fit(X, y)
w = reg.coef_  #reg coeff h1 - hn
c=reg.intercept_ # bias h0

xpoints = np.linspace(4,6)
ypoints = w[0] * xpoints + c;
plt.plot(xpoints, ypoints, 'g-')
plt.scatter(X, y,s=10)
plt.suptitle('Linear Regression IRIS Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

yPredict = reg.predict(X)
rmse = (np.sqrt(mean_squared_error(y, yPredict)))
r2 = r2_score(y, yPredict) #
print('Train RMSE =', rmse)
print('Train R2 score =', r2)
print("\n")