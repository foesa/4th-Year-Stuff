import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

data = {'x': [-1, 0, 1], 'y': [0, 1, 0]}
df = pd.DataFrame(data)
gamma = 25
gamma_Vals = [0, 1, 5, 10, 25]
c_vals = np.linspace(10, 100, num=10)
grid = np.linspace(start=-3, stop=3, num=100).reshape(-1, 1)


def gaussian_kernel(distances):
    weights = np.exp(- gamma * (distances ** 2))
    return weights / np.sum(weights)


def read_data():
    f = open("week6.txt", "r")
    start = True
    data = {'x': [], 'y': []}
    for i in f:
        if not start:
            i = i.rstrip('\n')
            vals = i.split(",")
            data['x'].append(float(vals[0]))
            data['y'].append(float(vals[1]))
        else:
            start = False
    return pd.DataFrame(data)


def KNGaus():
    for i in range(len(gamma_Vals)):
        global gamma
        gamma = gamma_Vals[i]
        model = KNeighborsRegressor(n_neighbors=3, weights=gaussian_kernel)
        model.fit(np.array(df['x']).reshape(-1, 1), df['y'])
        ys = model.predict(grid)
        plt.clf()
        plt.scatter(df['x'], df['y'], color='red', marker='+', label='Training Data')
        plt.plot(grid, ys, color='blue', label='Predictions')
        plt.xlabel("Input X")
        plt.ylabel("Output Y")
        plt.title(f'Predictions vs Training data, Gamma={gamma}')
        plt.legend()
        plt.show()


def kernelRidge():
    for i in range(len(c_vals)):
        for s in range(len(gamma_Vals)):
            global gamma
            gamma = gamma_Vals[s]
            model = KernelRidge(alpha=1 / (2 * c_vals[i]), kernel='rbf', gamma=gamma)
            model.fit(np.array(df['x']).reshape(-1, 1), df['y'])
            ys = model.predict(grid)
            plt.clf()
            plt.scatter(df['x'], df['y'], color='red', marker='+', label='Training Data')
            plt.plot(grid, ys, color='blue', label='Predictions')
            plt.xlabel("Input X")
            plt.ylabel("Output Y")
            plt.title(f'Predictions vs Training data, Gamma={gamma}, C={c_vals[i]}')
            plt.legend()
            plt.show()
            print(model.dual_coef_)


def hyper_pick_knn():
    alt_gamma_vals = np.linspace(0, 25, num=15)
    dataframe = read_data()
    mean_list = []
    variance_list = []
    for i in alt_gamma_vals:
        kf = KFold(n_splits=10)
        error_list = []
        global gamma
        gamma = i
        for train, test in kf.split(dataframe):
            x_train, x_test = dataframe.loc[train, 'x'], dataframe.loc[test, 'x']
            y_train, y_test = dataframe.loc[train, 'y'], dataframe.loc[test, 'y']
            model = KNeighborsRegressor(n_neighbors=len(x_train), weights=gaussian_kernel)
            model.fit(np.array(x_train).reshape(-1, 1), y_train)
            ys = model.predict(np.array(x_test).reshape(-1, 1))
            error_list.append(mean_squared_error(y_test.values, ys))
        error_list = np.array(error_list)
        mean_list.append(error_list.mean())
        variance_list.append(error_list.var())
    plt.clf()
    plt.errorbar(alt_gamma_vals, mean_list, variance_list, label="Mean Error", color='red', ecolor='black')
    plt.xlabel('Gamma Value')
    plt.ylabel('Mean Error')
    plt.legend()
    plt.show()

def hyper_pick_ridge_gamma():
    alt_gamma_vals = np.linspace(0, 25, num=25)
    dataframe = read_data()
    mean_list = []
    variance_list = []
    for i in c_vals:
        kf = KFold(n_splits=10)
        error_list = []
        # global gamma
        # gamma = i
        for train, test in kf.split(dataframe):
            x_train, x_test = dataframe.loc[train, 'x'], dataframe.loc[test, 'x']
            y_train, y_test = dataframe.loc[train, 'y'], dataframe.loc[test, 'y']
            model = KernelRidge(alpha=1/(2*i), kernel='rbf', gamma=4.16667)
            model.fit(np.array(x_train).reshape(-1, 1), y_train)
            ys = model.predict(np.array(x_test).reshape(-1, 1))
            error_list.append(mean_squared_error(y_test.values, ys))
        error_list = np.array(error_list)
        mean_list.append(error_list.mean())
        variance_list.append(error_list.var())

    plt.clf()
    plt.errorbar(list(map(lambda x: 1/(2*x), c_vals)), mean_list, variance_list, label="Mean Error", color='red', ecolor='black')
    plt.xlabel('Gamma Value')
    plt.ylabel('Mean Error')
    plt.legend()
    plt.show()
    print(c_vals)
    print(mean_list)

#Best Gamma = 4.17
# C = 30 best value of C

hyper_pick_ridge_gamma()
