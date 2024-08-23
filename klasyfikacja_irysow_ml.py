# Importowanie niezbędnych bibliotek
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

# Przygotowanie danych
# Wczytywanie danych z pliku CSV 'Iris.csv' za pomocą biblioteki Pandas
df = pd.read_csv('Iris.csv')
df.head()

# Ładowanie zbioru danych Iris z biblioteki scikit-learn
iris = load_iris()
data = iris.data
target = iris.target # Wczytaj etykiety z danych Iris
target_names = iris.target_names # Nazwy etykiet

# Podział danych na cechy (X) i etykiety (y)
X = iris.data  # Cechy (długość działki kielicha, szerokość działki kielicha, długość płatka, szerokość płatka)
y = iris.target  # Etykiety (0: setosa, 1: versicolor, 2: virginica)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Podziel dane na zbiory treningowy i testowy (80% treningowy, 20% testowy)

# Wybór modeli klasyfikacji
models = {
    'K-nearest neighbors (KNN)': KNeighborsClassifier(),
    'Support Vector Machines (SVM)': SVC(),
    'Random Forests': RandomForestClassifier()
}

# Dla każdego modelu, zdefiniuj różne zestawy hiperparametrów do przetestowania
param_grids = {
    'K-nearest neighbors (KNN)': {'n_neighbors': [3, 5, 7]},
    'Support Vector Machines (SVM)': {'C': [1, 10, 100], 'kernel': ['linear', 'rbf']},
    'Random Forests': {'n_estimators': [50, 100, 200]}
}


# Dla każdego modelu, testuj różne kombinacje hiperparametrów
best_models = {}
best_accuracies = {}

for model_name, model in models.items():
    param_grid = param_grids.get(model_name, {})  # Pobierz zestaw hiperparametrów dla danego modelu
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_accuracy = grid_search.best_score_

    best_models[model_name] = best_model
    best_accuracies[model_name] = best_accuracy

# Wyświetl najlepsze modele i ich dokładności
for model_name, best_model in best_models.items():
    print(f"Model: {model_name}")
    print(f"Najlepsze hiperparametry: {best_model.get_params()}")
    print(f"Cross-Validation Accuracy: {best_accuracies[model_name]:.2f}")

# Znajdź najlepszy model na podstawie dokładności
best_model_name = max(best_accuracies, key=best_accuracies.get)
best_model = best_models[best_model_name]
best_accuracy = best_accuracies[best_model_name]

print(f"Najlepszy model: {best_model_name}")
print(f"Najlepsze hiperparametry: {best_model.get_params()}")
print(f"Cross-Validation Accuracy: {best_accuracy:.2f}")

# Trenuj najlepszy model na całym zbiorze treningowym
best_model.fit(X_train, y_train)

# Wykonaj predykcje na zbiorze testowym
y_pred = best_model.predict(X_test)

# Wyświetlenie wyników
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print("Raport Klasyfikacji:")
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print(class_report)

# Tworzenie wykresu parowego do wizualizacji związków między cechami
sns.set(style="ticks")
sns.pairplot(sns.load_dataset("iris"), hue="species", markers=["o", "s", "D"])
plt.show()

# Importowanie bibliotek do wizualizacji granic decyzyjnych
from matplotlib.colors import ListedColormap

# Wybór dwóch cech do wizualizacji (np. długość działki kielicha i długość płatka)
X_train_2d = X_train[:, [0, 2]]
X_test_2d = X_test[:, [0, 2]]

# Tworzenie siatki punktów
h = 0.02  # Rozmiar kroku w siatce
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Trenowanie najlepszego modelu na dwóch cechach
if best_model_name == 'K-nearest neighbors (KNN)':
    best_knn = KNeighborsClassifier(**best_model.get_params())
    best_knn.fit(X_train_2d, y_train)
    Z = best_knn.predict(np.c_[xx.ravel(), yy.ravel()])
elif best_model_name == 'Support Vector Machines (SVM)':
    best_svm = SVC(**best_model.get_params())
    best_svm.fit(X_train_2d, y_train)
    Z = best_svm.predict(np.c_[xx.ravel(), yy.ravel()])
elif best_model_name == 'Random Forests':
    best_rf = RandomForestClassifier(**best_model.get_params())
    best_rf.fit(X_train_2d, y_train)
    Z = best_rf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

# Tworzenie wykresu konturowego
plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']), alpha=0.8)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap='Dark2', edgecolor='k')
plt.xlabel('Długość działki kielicha')
plt.ylabel('Długość płatka')
plt.title('Wizualizacja granicy decyzyjnej')
plt.show()
