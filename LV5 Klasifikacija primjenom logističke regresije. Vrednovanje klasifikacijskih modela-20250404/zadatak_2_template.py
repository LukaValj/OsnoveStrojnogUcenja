import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)



#a
train_class_counts = np.unique(y_train, return_counts=True)
test_class_counts = np.unique(y_test, return_counts=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].bar(train_class_counts[0], train_class_counts[1], color='lightblue')
ax[0].set_title("Broj primjera u trening skupu")
ax[0].set_xlabel("Vrsta pingvina")
ax[0].set_ylabel("Broj primjera")
ax[0].set_xticks(train_class_counts[0])
ax[0].set_xticklabels([labels[i] for i in train_class_counts[0]])

ax[1].bar(test_class_counts[0], test_class_counts[1], color='lightgreen')
ax[1].set_title("Broj primjera u test skupu")
ax[1].set_xlabel("Vrsta pingvina")
ax[1].set_ylabel("Broj primjera")
ax[1].set_xticks(test_class_counts[0])
ax[1].set_xticklabels([labels[i] for i in test_class_counts[0]])

plt.tight_layout()
plt.show()


#b
model = LogisticRegression(solver='lbfgs')

model.fit(X_train, y_train.ravel())  # ravel() u jednodimenzionalni niz

print("Koeficijenti modela (theta):\n", model.coef_)
print("Presjek modela (bias):", model.intercept_)



#c i d
plot_decision_regions(X_train, y_train, classifier=model)
plt.title("Granice odluke za model logističke regresije")
plt.xlabel('Dužina kljuna (mm)')
plt.ylabel('Dužina peraja (mm)')
plt.legend(loc='upper left')
plt.show()

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=labels.values())

print("Matrica zabune:")
print(cm)
print(f"\nTočnost modela: {accuracy:.4f}")
print("\nIzvještaj o klasifikaciji:")
print(report)

#f

input_variables = ['bill_length_mm', 'flipper_length_mm', 'bill_depth_mm', 'body_mass_g']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train.ravel())  # ravel() 1D niz

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=labels.values())

print("Matrica zabune:")
print(cm)
print(f"\nTočnost modela: {accuracy:.4f}")
print("\nIzvještaj o klasifikaciji:")
print(report)