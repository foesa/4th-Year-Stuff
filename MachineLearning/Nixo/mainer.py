import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from logger import logisticRegression

url = "Bluetooth_distance.csv"
df = pd.read_csv(url, header=None)

# question 1
blueTooth_signals = df.iloc[:, 1:].values
mean_of_bluetooth_signals = blueTooth_signals.mean(axis=1)
indicator_variable_for_distance = df.iloc[:, :1]

# plt.scatter(mean_of_bluetooth_signals, indicator_variable_for_distance, cmap='rainbow')
plt.title("scatter plot of the distance vs the average signal strength")
# plt.show()

# test
# bc = datasets.load_breast_cancer()
# print(bc.data), print(bc.data.shape)
# print(bc.target), print(bc.target.shape)
# X = bc.data
# y = bc.target

# question 2

X = blueTooth_signals
y = indicator_variable_for_distance.values  # target
new_array = np.array(y.flatten())  # target(flattened)

# normalise target
for x in range(len(new_array)):
    if new_array[x] == -1:
        new_array[x] = 0

# print(new_array)
# print(X), print(y), print(new_array), print(X.shape), print(y.shape), print(new_array.shape)

X_train, X_test, y_train, y_test = train_test_split(X, new_array, random_state=1)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy



model = LogisticRegression(penalty='none', solver='lbfgs')
model.fit(X, y)
ys = model.predict(X)
plt.scatter(mean_of_bluetooth_signals, ys)
plt.show()

regressor = logisticRegression(lr=0.0001, n_iters=1000)
regressor.fit(X, y)
predictions, _ = regressor.predict(X_test)

print("LR classification accuracy", accuracy(y_test, predictions))

y_axis, x_axis = regressor.predict(X)
w, b, l = regressor.the_goods()




plt.plot(x_axis, y_axis)
plt.show()

# plt.scatter(mean_of_bluetooth_signals, new_array, cmap='rainbow')
# plt.show()
# # sns.regplot(x=mean_of_bluetooth_signals, y=indicator_variable_for_distance, data=df, logistic=True)