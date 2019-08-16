import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.animation as animation
dataset = pd.read_csv('dataset.txt',sep='\t',header=None)
dataset.columns = ["X","y"]

x = dataset.iloc[:, 0].values
y = dataset.iloc[:,1].values
x2,y2 = x,y
plt.scatter(x,y)
plt.show()
x = np.reshape(x, (6, 1))
y = np.reshape(y,(6,1))
theta = np.array([1,0])
theta = np.reshape(theta,(2,1))
x1 = np.ones((6,2))
x1[:,1:] = x
m = len(x1)
alpha = 0.000212
plt.ion()
fig = plt.figure()
# fig.ylim([0, (max(y) + 30)])
# fig.scatter(x,y)
ax = fig.add_subplot(111)
line1, = ax.plot(x,y,'r-')


while True:
    y1 = theta[0]+theta[1]*x
    plt.scatter(x,y,c='b')
    line1.set_ydata(y1)
    fig.canvas.draw()

    h = x1.dot(theta)
    error = h-y
    sqrd_error = np.square(error)
    sum_sqrd_error = np.sum(sqrd_error)
    cost = (sum_sqrd_error/(2*m))
    xT = x1.T
    grad = (xT.dot(error))/(m)
    theta = theta - alpha*(grad)
    fig.canvas.flush_events()
