import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

dataset = pd.read_csv('../data/Data.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

Y = Y.reshape(len(Y),1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

X_sc = StandardScaler()
Y_sc = StandardScaler()

X_train = X_sc.fit_transform(X_train)
Y_train = Y_sc.fit_transform(Y_train)

regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, Y_train)

Y_pred = Y_sc.inverse_transform(regressor.predict(X_sc.transform(X_test)).reshape(-1,1))

np.set_printoptions(precision=2)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))

print(r2_score(Y_test, Y_pred))