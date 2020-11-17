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
c_vals = np.linspace(start=.1, stop=1000, num=50)
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
    dataframe = read_data()
    for i in range(len(gamma_Vals)):
        global gamma
        gamma = gamma_Vals[i]
        model = KNeighborsRegressor(n_neighbors=len(dataframe), weights=gaussian_kernel)
        model.fit(np.array(dataframe['x']).reshape(-1, 1), dataframe['y'])
        ys = model.predict(grid)
        plt.clf()
        plt.scatter(dataframe['x'], dataframe['y'], color='red', marker='+', label='Training Data')
        plt.plot(grid, ys, color='blue', label='Predictions')
        plt.xlabel("Input X")
        plt.ylabel("Output Y")
        plt.title(f'Predictions vs Training data, Gamma={gamma}')
        plt.legend()
        # plt.savefig(f'ks{i}.png', dpi=300, bbox_inches='tight')
        plt.show()


def kernel_Ridge():
    dataframe = read_data()
    # for i in range(len(c_vals)):
    for s in range(len(gamma_Vals)):
            global gamma
            gamma = gamma_Vals[s]
            model = KernelRidge(alpha=1 / (2 * 11.102), kernel='rbf', gamma=gamma)
            model.fit(np.array(dataframe['x']).reshape(-1, 1), dataframe['y'])
            ys = model.predict(grid)
            plt.clf()
            plt.scatter(dataframe['x'], dataframe['y'], color='red', marker='+', label='Training Data')
            plt.plot(grid, ys, color='blue', label='Predictions')
            plt.xlabel("Input X")
            plt.ylabel("Output Y")
            plt.title(f'Predictions vs Training data, Gamma={gamma}, C = {11.102}')
            plt.legend()
            plt.savefig(f'kz{s}.png', dpi=300, bbox_inches='tight')
            plt.show()
            # print(
            #     f'\(\\thetha_0 {round(model.dual_coef_[0], 9)}, thetha_1 {round(model.dual_coef_[1], 9)} , thetha_2 {round(model.dual_coef_[2], 9)}\)')


def hyper_pick_knn():
    alt_gamma_vals = np.linspace(0, 100)
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
    print(alt_gamma_vals)
    print(mean_list)


def hyper_pick_ridge_gamma():
    alt_gamma_vals = np.linspace(1, 5, num=25)
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
            model = KernelRidge(alpha=1/(2*11.102), kernel='rbf', gamma=i)
            model.fit(np.array(x_train).reshape(-1, 1), y_train)
            ys = model.predict(np.array(x_test).reshape(-1, 1))
            error_list.append(mean_squared_error(y_test.values, ys))
        error_list = np.array(error_list)
        mean_list.append(error_list.mean())
        variance_list.append(error_list.var())

    plt.clf()
    plt.errorbar(alt_gamma_vals, mean_list, variance_list, label="Mean Error", color='red',
                 ecolor='black')
    plt.xlabel('Gamma')
    plt.ylabel('Mean Error')
    plt.legend()
    plt.show()
    print(c_vals)
    print(mean_list)

def plotter():
    dataframe = read_data()
    global gamma
    gamma = 25
    model1 = KernelRidge(alpha=1/(2*11.102), kernel='rbf', gamma=1.8)
    model2 = KNeighborsRegressor(n_neighbors=len(dataframe), weights=gaussian_kernel)
    model1.fit(np.array(dataframe['x']).reshape(-1, 1), dataframe['y'])
    model2.fit(np.array(dataframe['x']).reshape(-1, 1), dataframe['y'])
    ys1 = model1.predict(np.array(grid).reshape(-1, 1))
    ys2 = model2.predict(np.array(grid).reshape(-1, 1))
    plt.clf()
    plt.scatter(dataframe['x'], dataframe['y'], color='green', marker='+', label='Training Data')
    plt.plot(grid, ys1, color='purple', label='Predictions(Kernel Ridge)')
    plt.plot(grid, ys2, color='red', label='Predictions(KNN)')
    plt.xlabel("Input X")
    plt.ylabel("Output Y")
    plt.title('Predictions vs Training data')
    plt.legend()
    plt.show()

plotter()