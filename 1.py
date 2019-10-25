import numpy as np

error = np.array([1, 2, 3, 4, 5])

cost = 0
for e in error:
    cost += e ** 2
cost /= len(error)

print(cost)
