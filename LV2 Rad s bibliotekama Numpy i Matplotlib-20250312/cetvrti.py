import numpy as np
import matplotlib.pyplot as plt
crni_kvadrat = np.zeros((50, 50))
bijeli_kvadrat = np.ones((50, 50))
red1 = np.hstack((crni_kvadrat, bijeli_kvadrat))
red2 = np.hstack((bijeli_kvadrat, crni_kvadrat))
slika = np.vstack((red1, red2))
plt.imshow(slika, cmap='gray', interpolation='nearest')
plt.show()