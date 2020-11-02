import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

f = open("ass41.txt", "r")
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
df1 = df[df['label'] == -1]
df2 = df[df['label'] != -1]


def graphs():
    plt.clf()
    plt.scatter(df1['x1'], df1['x2'], marker='+', color='red', label='y = -1')
    plt.scatter(df2['x1'], df2['x2'], marker='o', color='blue', label='y = 1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('X1 v X2 with labels for classes')
    plt.legend()
    plt.show()


def L1Logger():
    q_vals = [f for f in range(1, 7)]
    mean_list = []
    variance_list = []
    for i in q_vals:
        p = PolynomialFeatures(i).fit(df[['x1', 'x2']])
        features = pd.DataFrame(p.transform(df[['x1', 'x2']]), columns=p.get_feature_names(df.columns))
        kf = KFold(n_splits=10)
        error_list = []
        plotted = False
        for train, test in kf.split(features):
            x_train, x_test = features.loc[train], features.loc[test]
            y_train, y_test = df.loc[train, 'label'], df.loc[test, 'label']
            model = LogisticRegression(penalty='l1', solver='liblinear')
            model.fit(x_train.values, y_train.values)
            pred = model.predict(x_test)
            error_list.append(mean_squared_error(y_test.values, pred))
            if ((i == 1) or (i == 2) or (i == 6)) and not plotted:
                plt.clf()
                pred = model.predict(features)
                features['pred'] = pred
                i1 = features[features['pred'] == -1]
                i2 = features[features['pred'] != -1]
                plt.scatter(df1['x1'], df1['x2'], marker='+', color='red', label='Class:-1')
                plt.scatter(df2['x1'], df2['x2'], marker='+', color='blue', label='Class:1')
                plt.scatter(i1['x1'], i1['x2'], marker='o', color='green', label='Class:-1', facecolors='none')
                plt.scatter(i2['x1'], i2['x2'], marker='o', color='purple', label='Class:1', facecolors='none')
                plt.xlabel('X1')
                plt.ylabel('X2')
                plt.title(f'Q = {i}')
                plt.legend()
                plt.show()
                plotted = True
        error_list = np.array(error_list)
        mean = error_list.mean()
        mean_list.append(mean)
        var = error_list.var()
        variance_list.append(var)
    plt.clf()
    plt.errorbar(q_vals, mean_list, variance_list, label='Mean Error Line', color='red')
    plt.hlines(y=0, xmin=1, xmax=6, label='Training Data')
    plt.xlabel('Q value')
    plt.ylabel('Mean Error')
    plt.legend()
    plt.show()


def c_val():
    p = PolynomialFeatures(2).fit(df[['x1', 'x2']])
    features = pd.DataFrame(p.transform(df[['x1', 'x2']]), columns=p.get_feature_names(df.columns))
    c_vals = np.linspace(0.001, 0.2, num=25)
    mean_list = []
    std_list = []
    kf = KFold(n_splits=10)

    for i in c_vals:
        error_list = []
        printed = True
        for train, test in kf.split(df):
            x_train, x_test = features.loc[train], features.loc[test]
            y_train, y_test = df.loc[train, 'label'], df.loc[test, 'label']
            model = LogisticRegression(penalty='l1', solver='liblinear', C=i)
            model.fit(x_train, y_train)
            if printed:
                print(model.intercept_, model.coef_)
                printed = False
            pred = model.predict(x_test)
            error_list.append(mean_squared_error(y_test, pred))
        error_list = np.array(error_list)
        mean = error_list.mean()
        mean_list.append(mean)
        std = error_list.std()
        std_list.append(std)
    plt.clf()
    plt.errorbar(c_vals, mean_list, yerr=std_list)
    plt.xlabel('C Value')
    plt.ylabel('Mean Error')
    plt.title('C vs Mean Error')
    plt.show()


def Knn():
    mean_list = []
    std_list = []
    kf = KFold(n_splits=10)
    k_vals = [x for x in range(11) if x % 2 != 0]
    # p = PolynomialFeatures(2).fit(df[['x1', 'x2']])
    # features = pd.DataFrame(p.transform(df[['x1', 'x2']]), columns=p.get_feature_names(df.columns))
    for k in k_vals:
        error_list = []
        for train, test in kf.split(df):
            x_train, x_test = df.loc[train, ['x1','x2']], df.loc[test, ['x1','x2']]
            y_train, y_test = df.loc[train, 'label'], df.loc[test, 'label']
            model = KNeighborsClassifier(n_neighbors=k, weights='uniform').fit(x_train, y_train)
            pred = model.predict(x_test)
            error_list.append(mean_squared_error(y_test, pred))
        error_list = np.array(error_list)
        mean = error_list.mean()
        mean_list.append(mean)
        std = error_list.std()
        std_list.append(std)
    plt.clf()
    plt.errorbar(k_vals, mean_list, yerr=std_list)
    plt.xlabel('K')
    plt.ylabel('Mean Error')
    plt.title('K vs Mean Error')
    plt.show()


L1Logger()