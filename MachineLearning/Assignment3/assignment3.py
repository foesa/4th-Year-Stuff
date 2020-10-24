import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

f = open("ass3.txt", "r")
start = True
data = {'x1': [], 'x2': [], 'label': []}
for i in f:
    if not start:
        i = i.rstrip('\n')
        vals = i.split(",")
        data['x1'].append(float(vals[0]))
        data['x2'].append(float(vals[1]))
        data['label'].append(float(vals[2]))
    else:
        start = False
df = pd.DataFrame(data)


def graps():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['x1'], df['x2'], df['label'])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Target')
    plt.show()
    plt.clf()
    plt.scatter(df['x2'], df['label'])
    plt.xlabel('x2')
    plt.ylabel('Target')
    plt.show()


def Lassor():
    p = PolynomialFeatures(5).fit(df[['x1', 'x2']])
    features = pd.DataFrame(p.transform(df[['x1', 'x2']]), columns=p.get_feature_names(df.columns))
    models = []
    c_vals = [1e-7, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    for s in c_vals:
        model = Lasso(alpha=s)
        model.fit(features, df['label'])
        models.append((model, s))

    x1vals = y1vals = np.array(np.linspace(-2, 2))
    x, y = np.meshgrid(x1vals, y1vals)
    positions = np.vstack([x.ravel(), y.ravel()])
    xtest = (np.array(positions)).T
    pdata = pd.DataFrame(xtest, columns=['x1', 'x2'])
    p1 = PolynomialFeatures(5).fit(pdata[['x1', 'x2']])
    mesh_features = pd.DataFrame(p1.transform(pdata[['x1', 'x2']]), columns=p.get_feature_names(pdata.columns))
    for i in models:
        pred = i[0].predict(mesh_features)
        pred = pred.reshape(x.shape)
        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['x1'], df['x2'], df['label'], marker='+', color='red', s=100)
        ax.plot_surface(x, y, pred, alpha=0.5)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Target')
        plt.title(f'C = {i[1]}')
        plt.show()


def cross_val():
    k_vals = [2, 5, 10, 25, 50, 100]
    mean_list = []
    variance_list = []
    for k in k_vals:
        error_list = []
        kf = KFold(n_splits=k)
        for train, test in kf.split(df):
            # print('Train', train)
            # print('Test', test)
            x_train, x_test = df.loc[train, ['x1', 'x2']], df.loc[test, ['x1', 'x2']]
            y_train, y_test = df.loc[train, 'label'], df.loc[test, 'label']
            model = Lasso(alpha=1)
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            error_list.append(mean_squared_error(y_test, pred))
        error_list = np.array(error_list)
        mean = error_list.mean()
        mean_list.append(mean)
        var = error_list.var()
        variance_list.append(var)
        print(mean, var)
    plt.clf()
    plt.errorbar(k_vals, mean_list, yerr=variance_list)
    plt.xlabel('Folds')
    plt.ylabel('Mean Error')
    plt.title('Folds vs Mean Error')
    plt.show()

def c_pick():
    c_vals = np.linspace(0.00000001,0.4)
    mean_list = []
    std_list = []
    kf = KFold(n_splits=10)
    for i in c_vals:
        error_list = []
        for train, test in kf.split(df):
            # print('Train', train)
            # print('Test', test)
            x_train, x_test = df.loc[train, ['x1', 'x2']], df.loc[test, ['x1', 'x2']]
            y_train, y_test = df.loc[train, 'label'], df.loc[test, 'label']
            model = Lasso(alpha=i)
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            error_list.append(mean_squared_error(y_test, pred))
        error_list = np.array(error_list)
        mean = error_list.mean()
        mean_list.append(mean)
        std = error_list.std()
        std_list.append(std)
    plt.clf()
    plt.errorbar(c_vals, mean_list, yerr=std_list)
    plt.xlabel('Folds')
    plt.ylabel('Mean Error')
    plt.title('Folds vs Mean Error')
    plt.show()




c_pick()