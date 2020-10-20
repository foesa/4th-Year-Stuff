import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Reads in the data and adds it to a hashmap with 2 lists; a list of inputs (x) and outputs (y)
f = open("ass2.txt", "r")
start = True
data = {'x': [], 'y': [], 'z': []}
for i in f:
    if not start:
        i = i.rstrip('\n')
        vals = i.split(",")
        data['x'].append(float(vals[0]))
        data['y'].append(float(vals[1]))
        data['z'].append(int(vals[2]))
    else:
        start = False
df = pd.DataFrame(data)



def loger():
    df1 = df[df['z'] == -1]
    df2 = df[df['z'] != -1]
    x = df[['x', 'y']]
    model = LogisticRegression(penalty='none', solver='lbfgs')
    model.fit(x, df['z'])
    print(model.intercept_, model.coef_)
    ys = model.predict(x)
    x['ys'] = ys
    df3 = x[x['ys'] == -1]
    df4 = x[x['ys'] != -1]

    x_vals = np.linspace(-1, 1, 50)
    y = -(model.intercept_ + model.coef_[0][0] * x_vals) / model.coef_[0][1]
    plt.scatter(df1['x'], df1['y'], marker='+')
    plt.scatter(df2['x'], df2['y'], marker='o')
    plt.scatter(df3['x'], df3['y'], marker='+', color='green')
    plt.scatter(df4['x'], df4['y'], marker='o', color='purple')
    plt.plot(x_vals,y, color='black')
    plt.legend(["Decision Boundary","L_1", "L_2", "L_1(pred)", "L_2(pred)"])
    plt.title(f"Training Data")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def svm():
    df1 = df[df['z'] == -1]
    df2 = df[df['z'] != -1]
    x = df[['x', 'y']]
    models = []
    c_vals = np.geomspace(.001, 1000, num=7)
    print(c_vals)
    for i in c_vals:
        model = LinearSVC(C=i)
        model.fit(x, df['z'])
        models.append((model,i))

    for model in models:
        plt.clf()
        model[0].fit(x,df['z'])
        ys = model[0].predict(x)
        x['ys'] = ys
        df3 = x[x['ys'] == -1]
        df4 = x[x['ys'] != -1]
        x_vals = np.linspace(-1, 1, 50)
        y = -(model[0].intercept_ + model[0].coef_[0][0] * x_vals) / model[0].coef_[0][1]
        plt.scatter(df1['x'], df1['y'], marker='+')
        plt.scatter(df2['x'], df2['y'], marker='o')
        plt.scatter(df3['x'], df3['y'], marker='+', color='green')
        plt.scatter(df4['x'], df4['y'], marker='o', color='purple')
        plt.plot(x_vals, y, color='black')
        plt.legend(["Decision Boundary", "L_1", "L_2", "L_1(pred)", "L_2(pred)" ])
        plt.title(f"C= {model[1]}: Training Data v Predictions")
        plt.xlabel("x_1")
        plt.ylabel("x_2")
        plt.show()
        plt.clf()

def squared():
    plt.clf()
    df['x3'] = df['x']**2
    df['x4'] = df['y']**2
    df1 = df[df['z'] == -1]
    df2 = df[df['z'] != -1]
    x = df[['x', 'y']]
    square = df[['x', 'y', 'x3', 'x4']]
    model = LogisticRegression(penalty='none', solver='lbfgs')
    model.fit(square, df['z'])
    print(model.intercept_, model.coef_)
    ys = model.predict(square)
    x['ys'] = ys
    df3 = x[x['ys'] == -1]
    df4 = x[x['ys'] != -1]
    plt.scatter(df1['x'], df1['y'], marker='+')
    plt.scatter(df2['x'], df2['y'], marker='o')
    plt.scatter(df3['x'], df3['y'], marker='+', color='green')
    plt.scatter(df4['x'], df4['y'], marker='o', color='purple')
    plt.legend(["L_1", "L_2", "L_1(pred)", "L_2(pred)"])
    plt.title(f"X1 v X2, Pred v Training Data")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    v = np.random.randint(10,size=999)
    for i in range(len(v)):
        if v[i] >= 7:
            v[i] = -1
        else:
            v[i] = -1
    df['rand'] = v
    df5 = df[df['rand'] == -1]
    df6 = df[df['rand'] != -1]
    plt.clf()
    plt.scatter(df1['x'], df1['y'], marker='+')
    plt.scatter(df2['x'], df2['y'], marker='o')
    plt.scatter(df5['x'], df5['y'], marker='+', color='green')
    plt.scatter(df6['x'], df6['y'], marker='o', color='purple')
    plt.show()
    num = df[df['rand'] == df['z']]
    num2 = df[df['z'] == ys]
    print(len(num), len(num2))

squared()