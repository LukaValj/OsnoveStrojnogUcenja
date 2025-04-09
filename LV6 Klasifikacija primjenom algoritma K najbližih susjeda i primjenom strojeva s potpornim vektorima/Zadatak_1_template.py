import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

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
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()



#1
# KNN model s K=5
KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train_n, y_train)

y_train_p_knn = KNN_model.predict(X_train_n)
y_test_p_knn = KNN_model.predict(X_test_n)

print("\nKNN (K=5): ")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p_knn)))
print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_p_knn)))

print("\nUsporedba rezultata:")
print("Logistička regresija - Točnost train: ", "{:0.3f}".format(accuracy_score(y_train, y_train_p)))
print("Logistička regresija - Točnost test: ", "{:0.3f}".format(accuracy_score(y_test, y_test_p)))
print("KNN (K=5) - Točnost train: ", "{:0.3f}".format(accuracy_score(y_train, y_train_p_knn)))
print("KNN (K=5) - Točnost test: ", "{:0.3f}".format(accuracy_score(y_test, y_test_p_knn)))

plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (K=5) - Točnost: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p_knn)))
plt.tight_layout()
plt.show()

#2
# Granica odluke za KNN model s K=1
KNN_model_1 = KNeighborsClassifier(n_neighbors=1)
KNN_model_1.fit(X_train_n, y_train)

plot_decision_regions(X_train_n, y_train, classifier=KNN_model_1)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (K=1) - Točnost: " + "{:0.3f}".format(accuracy_score(y_train, KNN_model_1.predict(X_train_n))))
plt.tight_layout()
plt.show()

# Granica odluke za KNN model s K=100
KNN_model_100 = KNeighborsClassifier(n_neighbors=100)
KNN_model_100.fit(X_train_n, y_train)

plot_decision_regions(X_train_n, y_train, classifier=KNN_model_100)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (K=100) - Točnost: " + "{:0.3f}".format(accuracy_score(y_train, KNN_model_100.predict(X_train_n))))
plt.tight_layout()
plt.show()

#najbolji je ovaj sa K=5


##########################
#Drugi zadatak, unakrstna validacija
param_grid={'n_neighbors':np.arange(1,21)}

knn_cv=GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
knn_cv.fit(X_train_n, y_train)

print("\nUnakrsna validacija - odabir najboljeg K:")
print(f"Najbolji K: {knn_cv.best_params_['n_neighbors']}") #optimalna vrijednost K prema validaciji
print(f"Najbolja prosječna točnost (cv): {knn_cv.best_score_:.3f}") #prosječna tocnost na 5-fold cross-validaciji
#najbolji k je 7

##########################
#Treći zadatak, SVM model koji koristi RBF kernel funkciju

C_value=10
gamma_value=0.5

svm_rbf=svm.SVC(C=C_value, kernel='rbf', gamma=gamma_value)     #kernel: rbf, linear, poly, sigmoid
svm_rbf.fit(X_train_n, y_train)

y_test_pred=svm_rbf.predict(X_test_n)
test_acc= accuracy_score( y_test, y_test_pred)

print(f"\nSVM (C={C_value}, gamma={gamma_value})")
print(f"Točnost na test skupu: {test_acc:.3f}")

plot_decision_regions(X_train_n, y_train, classifier=svm_rbf)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title(f'SVM \nC={C_value}, gamma={gamma_value}\nTočnost test: {test_acc:.3f}')
plt.tight_layout()
plt.show()



##########################
#četvrti, optimalna vrijednost hiperparametra C i γ
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1, 10]
}
svm=svm.SVC(kernel='rbf')
grid_svm=GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy')
grid_svm.fit(X_train_n,y_train)

print("\nOptimalni hiperparametri (SVM RBF):")
print(f"Najbolji C: {grid_svm.best_params_['C']}")
print(f"Najbolji gamma: {grid_svm.best_params_['gamma']}")
print(f"Najbolja prosječna točnost (cv): {grid_svm.best_score_:.3f}")

best_svm_model = grid_svm.best_estimator_
plot_decision_regions(X_train_n, y_train, classifier=best_svm_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title(f'SVM RBF kernel\nC={grid_svm.best_params_["C"]}, gamma={grid_svm.best_params_["gamma"]}\nTočnost: {grid_svm.best_score_:.3f}')
plt.tight_layout()
plt.show()
#najbolji je 1, 1