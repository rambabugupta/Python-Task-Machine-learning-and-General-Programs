import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'],1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
clr = neighbors.KNeighborsClassifier()
clr.fit(X_train, y_train)
accuracy = clr.score(X_test,y_test)

print(accuracy)

predict_examples = np.array([[1,2,3,4,2,3,4,2,1],[1,2,3,4,2,1,2,2,1]])
predict_examples=predict_examples.reshape(len(predict_examples),-1)

prediction=clr.predict(predict_examples)

print(prediction)