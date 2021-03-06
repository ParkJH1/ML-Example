import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_data = np.array([8, 24, 28, 46, 44, 55, 79, 80, 99, 105])

slope = 10.715151515151513
intercept = -2.133333333333353  # y = slope * x + intercept

linear_regression_y = slope * x_data + intercept

plt.figure(0)
plt.plot(x_data, y_data, 'ro', label='y_data')
plt.plot(x_data, linear_regression_y, 'b-', linewidth=3, label='linear regression')
plt.legend()
plt.show()
