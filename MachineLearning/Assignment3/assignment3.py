import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
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
    c_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    for s in c_vals:
        model = Lasso(alpha=s)
        model.fit(features, df['label'])
        models.append((model, s))

    x1vals= y1vals = np.array(np.linspace(-2,2))
    x,y = np.meshgrid(x1vals,y1vals)
    positions = np.vstack([x.ravel(),y.ravel()])
    xtest = (np.array(positions)).T
    pdata = pd.DataFrame(xtest, columns=['x1','x2'])
    p1 = PolynomialFeatures(5).fit(pdata[['x1','x2']])
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
        plt.show()


Lassor()
