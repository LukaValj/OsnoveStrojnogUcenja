import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.csv', delimiter=',', skiprows=1)

# a) Broj osoba na temelju veličine numpy polja
broj_osoba = data.shape[0]
print(f"Broj osoba: {broj_osoba}")

# b) Prikaz odnosa visine i mase svih osoba
plt.scatter(data[:, 1], data[:, 2], c='blue', label='Osobe', s=1)
plt.xlabel('Visina (cm)')
plt.ylabel('Masa (kg)')
plt.title('Odnos visine i mase (sve osobe)')
plt.legend()
plt.show()

# c) Prikaz odnosa visine i mase svake pedesete osobe
svaka_pedeseta = data[::50]
plt.scatter(svaka_pedeseta[:, 1], svaka_pedeseta[:, 2], c='red', label='Svaka pedeseta osoba', s=5)
plt.xlabel('Visina (cm)')
plt.ylabel('Masa (kg)')
plt.title('Odnos visine i mase (svaka pedeseta osoba)')
plt.legend()
plt.show()

# d) Izračun minimalne, maksimalne i srednje vrijednosti visine
min_visina = np.min(data[:, 1])
max_visina = np.max(data[:, 1])
srednja_visina = np.mean(data[:, 1])
print(f"Minimalna visina: {min_visina} cm")
print(f"Maksimalna visina: {max_visina} cm")
print(f"Srednja visina: {srednja_visina:.2f} cm")

# e) Ponovite zadatak pod d), ali samo za muškarce, odnosno žene.

muskarci = data[:,0] == 1
zene = data[:,0] == 0

visine_muski = data[:, 1][muskarci]
visine_zene = data[:, 1][zene]

print(f"Visina muski min: {visine_muski.min()} cm")
print(f"Visina muski max: {visine_muski.max()} cm")
print(f"Visina muski srednja: {visine_muski.mean()} cm")

print(f"Visina zene min: {visine_zene.min()} cm")
print(f"Visina zene max: {visine_zene.max()} cm")
print(f"Visina zene srednja: {visine_zene.mean()} cm")
