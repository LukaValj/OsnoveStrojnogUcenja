import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 3, 3, 2, 1])
y = np.array([1, 1, 2, 2, 1])

plt.figure()
plt.plot(x, y, 'r', marker='X', markersize = 20, linewidth = 5) 
plt.fill(x, y, color='skyblue')

plt.title("Trapez")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.axis('equal')
plt.xlim(0,4)
plt.ylim(0,4)
plt.show()
