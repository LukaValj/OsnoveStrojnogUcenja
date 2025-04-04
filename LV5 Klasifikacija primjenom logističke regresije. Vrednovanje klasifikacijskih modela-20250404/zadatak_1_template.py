import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#a
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

plt.figure(figsize=(8, 6))

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter', marker='o', label='učenje')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='magma', marker='x', label='testiranje')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Prikaz podataka za učenje i testiranje')
plt.legend()
plt.show()

#b i c
theta_0 = model.intercept_[0]
theta_1, theta_2 = model.coef_[0]
print("Paramenti modela:")
print(f"Intercept (θ0): {theta_0}")
print(f"Koeficijenti (θ1, θ2): {theta_1}, {theta_2}")

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', marker='o', label='Podaci za učenje')

#granica odluke
xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100),
                     np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100))
Z = theta_0 + theta_1 * xx + theta_2 * yy

plt.contour(xx, yy, Z, levels=[0], cmap="coolwarm", linewidths=2)

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Granica odluke logističke regresije')
plt.legend()
plt.show()
print("\n")

#d
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Matrica zabune:")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print(f"Točnost: {accuracy:.2f}")
precision = precision_score(y_test, y_pred)
print(f"Preciznost: {precision:.2f}")
recall = recall_score(y_test, y_pred)
print(f"Odziv: {recall:.2f}")
print("\n")

#e
plt.figure(figsize=(8, 6))
plt.scatter(X_test[y_test == y_pred, 0], X_test[y_test == y_pred, 1], c='green', marker='o', label='Dobro klasificirani')
plt.scatter(X_test[y_test != y_pred, 0], X_test[y_test != y_pred, 1], c='black', marker='x', label='Pogrešno klasificirani')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Testni podaci s označenim ispravnim i pogrešnim klasifikacijama')
plt.legend()
plt.show()