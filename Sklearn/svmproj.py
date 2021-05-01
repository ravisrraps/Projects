from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

iris = datasets.load_iris()

x = iris.data
y = iris.target


classes = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y)

model = svm.SVC()
model.fit(x_train, y_train)

print(model)

predictions = model.predict(x_test)
acc = accuracy_score(y_test, predictions)

print('Predictions: ',predictions)
print('actual: ',y_test)
print('accuracy: ',acc)


for i in range(len(predictions)):
    print(classes[predictions[i]])