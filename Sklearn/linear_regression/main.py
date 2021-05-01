from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

boston = datasets.load_boston()


x = boston.data
y = boston.target

print('x')
print(x)
print(x.shape)
print('y')
print(y)
print(y.shape)

l_reg = linear_model.LinearRegression()
 
plt.scatter(x.T[0], y)

plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = l_reg.fit(x_train, y_train)
predictions = model.predict(x_test)
print('Predictions: ', predictions)
print('R^2 value: ', l_reg.score(x, y))
print('coedd: ', l_reg.coef_)
print('intercept: ', l_reg.intercept_)