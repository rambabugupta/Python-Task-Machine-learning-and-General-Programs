df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'],1, inplace=True)

#filling balnk values by 0
df.fillna(0,inplace=True)

#Dropping columns, keeping lables and features
churn_feat_space = df.drop(['class'],1)
churn_result = df['class']

##Coverting into arrays

#Using numpy array
#X = np.array(df.drop(['class'],1))
#y = np.array(df['class'])

X = churn_feat_space.as_matrix().astype(np.float)
y = np.array(churn_result)

#Copy from here

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.3)
clf = RandomForestClassifier(n_estimators=30)
clf.fit(X_train, y_train)


score = clf.score(X_test, y_test)
print("Accuracy: ", score)

