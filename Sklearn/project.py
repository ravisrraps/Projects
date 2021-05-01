import numpy as np
import pandas as pd
from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('car.data')


x = data[['buying',
         'maint',
         'safety']].values
y = data[['class']]




Le = LabelEncoder()
for i in range(len(x[0])):
    x[:, i] = Le.fit_transform(x[:, i])


label_maping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y['class'] = y['class'].map(label_maping)
y = np.array(y)


knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print('predictions: ',prediction)
print('accuracy: ',accuracy)


a = int(input('Index Value'))
print('actual value: ',y[a])
print('predicted value: ',knn.predict(x)[a])


