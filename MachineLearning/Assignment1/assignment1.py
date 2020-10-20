import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Reads in the data and adds it to a hashmap with 2 lists; a list of inputs (x) and outputs (y)
f = open("ass1data.txt", "r")
start = True
data = {'x': [], 'y': []}
for i in f:
    if not start:
        i = i.rstrip('\n')
        vals = i.split(",")
        data['x'].append(int(vals[0]))
        data['y'].append(float(vals[1]))
    else:
        start = False
df = pd.DataFrame(data)

# Normalises the data use min - max for inputs and outputs, giving values between 0-1
minY = df['y'].min()
maxY = df['y'].max()
df["y"] = df["y"].apply(lambda x: (x - minY) / (maxY - minY))
minX = df['x'].min()
maxX = df['x'].max()
df['x'] = df['x'].apply(lambda x: (x - minX) / (maxX - minX))




#
def grad_descent():
    theta_0 = 0
    theta_1 = 1
    l_rate = 0.7
    m = len(df)
    iteration = 0
    data2 = {'x': [], 'y': []}

    pred = lambda x: (theta_0 + (theta_1 * x))
    cost_func = lambda x, y: (pred(x) - y) ** 2

    while iteration != 100:
        cost = 0
        temp0 = 0
        temp1 = 0
        for row in df.itertuples():
            cost = cost + cost_func(row.x, row.y)
            val = -2 / m * (row.y - pred(row.x))
            val1 = (-2 / m * row.x) * (row.y - pred(row.x))

            temp0 = temp0 + val
            temp1 = temp1 + val1
        data2['x'].append(iteration)
        data2['y'].append(cost)

        theta_0 = theta_0 - (l_rate * temp0)
        theta_1 = theta_1 - (l_rate * temp1)
        iteration = iteration + 1

    # plt.scatter(data2['x'], data2['y'])
    # plt.xlabel("Iterations")
    # plt.ylabel("J(θ0,θ1)")
    # plt.title("α= 0.005: Iterations v J(θ0,θ1)")
    # plt.show()
    print(theta_0, theta_1)
    return theta_0, theta_1, data2


t0, t1, cost = grad_descent()
dp = {'x': [], 'y': []}
for x in df['x']:
    dp['x'].append(x)
    dp['y'].append(t0 + (t1 * x))
x = df['x'].values
y = df['y'].values
model = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))

plt.plot(dp['x'], dp['y'], color='red', linewidth=3)
plt.plot(x, model.predict(x.reshape(-1, 1)), color='green', linewidth=3)
plt.scatter(df['x'], df['y'], color='black')
plt.yticks(np.arange(0, 1, 0.1))
plt.xticks(np.arange(0, 1, 0.1))
plt.xlabel("x-input (Normalized)")
plt.ylabel("y-output (Normalized)")
plt.title('Plot of Data points')
plt.legend(["Grad Desc","Sklearn", "Training Data"])
plt.show()
print(model.intercept_, model.coef_)
