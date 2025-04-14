import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_3.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255          ###Ako test_4 onda stavi #

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()
img_array_next = img_array.copy()
img_array_last = img_array.copy()

###
K = 4

km = KMeans(n_clusters=K, random_state=0)
km.fit(img_array_aprox)
labels = km.predict(img_array_aprox)
centroids = km.cluster_centers_

#centroidi
print("Centri klastera (RGB vrijednosti):")
for i, c in enumerate(centroids):
    print(f"Centar {i + 1}: R={c[0]:.3f}, G={c[1]:.3f}, B={c[2]:.3f}")

img_array_aprox[:, 0] = centroids[labels][:, 0]
img_array_aprox[:, 1] = centroids[labels][:, 1]
img_array_aprox[:, 2] = centroids[labels][:, 2]

img_array_aprox = np.reshape(img_array_aprox, (w, h, d))

plt.figure()
plt.title(f"Kvantizirana slika (K = {K})")
plt.imshow(img_array_aprox)
plt.axis('off')
plt.tight_layout()
plt.show()

### J naći K
J_values = []
K_range = range(2, 11)

for K in K_range:
    km = KMeans(n_clusters=K, random_state=0)
    km.fit(img_array_next)
    J_values.append(km.inertia_)

plt.figure()
plt.plot(K_range, J_values, marker='o')
plt.title("Ovisnost J o broju cluster K")
plt.xlabel("Broj cluster K")
plt.ylabel("J (inertia)")
plt.grid(True)
plt.tight_layout()
plt.show()


##binarnaslika
K = 4
km = KMeans(n_clusters=K, random_state=0)
km.fit(img_array_last)
labels = km.predict(img_array_last)

for k in range(K):
    binary_mask = (labels == k).astype(float)
    binary_image = np.reshape(binary_mask, (w, h))

    plt.figure()
    plt.title(f"Binarna slika cluster {k+1}")
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()