import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


def minumum_mean_square_error(y_predict, y_data):
    ret = 0
    for i in range(len(y_predict)):
        ret += (y_data[i] - y_predict[i]) ** 2
    ret /= len(y_predict)
    return ret


x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_data = np.array([8, 24, 28, 46, 44, 55, 79, 80, 99, 105])

# mse_x = np.arange(-15, 25)
mse_x = np.linspace(5, 15, 100)
# mse_y = np.arange(-15, 25)
mse_y = np.linspace(-7, 3, 100)
mse_x, mse_y = np.meshgrid(mse_x, mse_y)
mse_z = []

A = np.sum(np.square(x_data))
B = np.sum(x_data * y_data)
C = np.sum(y_data)
D = np.sum(x_data)

b = (B - C * A / D) / (D - 10 * A / D)
a = (C - 10 * b) / D
print(a, b)

p = np.polyfit(x_data, y_data, 1)
print(p[0], p[1])

for slope in mse_x[0]:
    tmp = []
    for intersect in mse_y:
        y_predict = slope * x_data + intersect[0]
        error = minumum_mean_square_error(y_predict, y_data)
        tmp.append(error)
        print(slope, intersect[0], error)
    mse_z.append(tmp)

mse_z = np.array(mse_z)

fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(mse_x, mse_y, mse_z)

plt.show()
