import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_data = np.array([8, 24, 28, 46, 44, 55, 79, 80, 99, 105])
fx_data = np.array([10 * x for x in x_data])

p = np.polyfit(x_data, y_data, 1)

plt.figure(0)
plt.plot(x_data, y_data, 'ro', label='y_data')
plt.plot(x_data, np.polyval(p, x_data), 'b-', linewidth=3, label='fx_data')
plt.plot(x_data, fx_data, 'y-', linewidth=3, label='fx_data')
plt.legend()
plt.show()
