import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def plot_roc_curve(fpr, tpr, auc_value):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Zdefiniuj callback do zapisywania najlepszego modelu
checkpoint_path = "najlepszy_model.h5"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',  # Monitoruj dokładność walidacji
    save_best_only=True,      # Zapisuj tylko najlepsze modele
    mode='max',               # Zapisuj, gdy dokładność walidacji osiągnie maksimum
    verbose=1
)

# Zdefiniuj callback EarlyStopping
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',   # Monitoruj stratę walidacji
    patience=3,            # Przerwij trening po 3 kolejnych epokach bez poprawy
    restore_best_weights=True  # Przywróć wagi do najlepszych przed przerwaniem
)
# Załaduj zbiór danych MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Znormalizuj wartości pikseli do przedziału od 0 do 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# -------------------
# Model 1: Głęboka sieć neuronowa (DNN)
model = tf.keras.models.Sequential(name="model_dnn")  # Przypisz nazwę modelowi
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_split=0.1, callbacks=[checkpoint_callback, early_stopping_callback])

# -------------------
# Model 2: Konwolucyjna sieć neuronowa (CNN)
x_train_cnn = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test_cnn = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train_cnn = x_train_cnn.astype('float32') / 255.0
x_test_cnn = x_test_cnn.astype('float32') / 255.0

model_cnn = tf.keras.models.Sequential(name="model_cnn")  # Przypisz nazwę modelowi
model_cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model_cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
model_cnn.add(tf.keras.layers.Flatten())
model_cnn.add(tf.keras.layers.Dense(128, activation='relu'))
model_cnn.add(tf.keras.layers.Dense(10, activation='softmax'))

model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_cnn.fit(x_train_cnn, y_train, epochs=5, validation_split=0.1, callbacks=[checkpoint_callback, early_stopping_callback])

# -------------------
# Model 3: Rekurencyjna sieć neuronowa (RNN)
model_rnn = tf.keras.models.Sequential(name="model_rnn")  # Przypisz nazwę modelowi
model_rnn.add(tf.keras.layers.SimpleRNN(128, input_shape=(28, 28)))
model_rnn.add(tf.keras.layers.Dense(10, activation='softmax'))

model_rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_rnn.fit(x_train_cnn, y_train, epochs=5, validation_split=0.1, callbacks=[checkpoint_callback, early_stopping_callback])

# -------------------
# Model 4: Głęboka sieć neuronowa z warstwą Dropout
model_dnn_dropout = tf.keras.models.Sequential(name="model_dnn_dropout")  # Przypisz nazwę modelowi
model_dnn_dropout.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model_dnn_dropout.add(tf.keras.layers.Dense(128, activation='relu'))
model_dnn_dropout.add(tf.keras.layers.Dropout(0.5))
model_dnn_dropout.add(tf.keras.layers.Dense(128, activation='relu'))
model_dnn_dropout.add(tf.keras.layers.Dropout(0.5))
model_dnn_dropout.add(tf.keras.layers.Dense(10, activation='softmax'))

model_dnn_dropout.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_dnn_dropout.fit(x_train, y_train, epochs=5, validation_split=0.1, callbacks=[checkpoint_callback, early_stopping_callback])

# -------------------
# Model 5: Głęboka sieć neuronowa z normalizacją wsadową
model_dnn_batchnorm = tf.keras.models.Sequential(name="model_dnn_batchnorm")  # Przypisz nazwę modelowi
model_dnn_batchnorm.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model_dnn_batchnorm.add(tf.keras.layers.Dense(128, activation='relu'))
model_dnn_batchnorm.add(tf.keras.layers.BatchNormalization())
model_dnn_batchnorm.add(tf.keras.layers.Dense(128, activation='relu'))
model_dnn_batchnorm.add(tf.keras.layers.BatchNormalization())
model_dnn_batchnorm.add(tf.keras.layers.Dense(10, activation='softmax'))

model_dnn_batchnorm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_dnn_batchnorm.fit(x_train, y_train, epochs=5, validation_split=0.1, callbacks=[checkpoint_callback, early_stopping_callback])

# -------------------
# Załaduj najlepszy model z pliku
najlepszy_model = tf.keras.models.load_model(checkpoint_path)
print(f"Nazwa najlepszego modelu: {najlepszy_model.name}")

# Ocena najlepszego modelu
strata, dokladnosc = najlepszy_model.evaluate(x_test, y_test)
print(f"Najlepszy Model - Strata: {strata}, Dokładność: {dokladnosc}")

#-----------
checkpoint_path = "najlepszy_model.h5"
najlepszy_model = tf.keras.models.load_model(checkpoint_path)
strata, dokladnosc = najlepszy_model.evaluate(x_test, y_test)
print(f"Najlepszy Model - Strata: {strata}, Dokładność: {dokladnosc}")

# Ocena dodatkowych metryk
y_pred = najlepszy_model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# One-hot encode y_test
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Oblicz precyzję, czułość i krzywą ROC
precision = precision_score(y_test, y_pred_labels, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_labels, average='weighted', zero_division=0)

# Oblicz krzywą ROC i AUC dla każdej klasy
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Oblicz krzywą ROC i AUC dla mikro-średniej
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Wyświetl dodatkowe metryki
print(f"Precyzja: {precision:.4f}")
print(f"Czułość: {recall:.4f}")

strata_przed, dokladnosc_przed = najlepszy_model.evaluate(x_test, y_test)
# Wyświetl krzywą ROC
plot_roc_curve(fpr["micro"], tpr["micro"], roc_auc["micro"])
# 1. Zmiana hiperparametrów
najlepszy_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 2. Dodanie regularyzacji
najlepszy_model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))

# 3. Batch Normalization
najlepszy_model.add(tf.keras.layers.BatchNormalization())

# 4. Zmiana funkcji aktywacji
najlepszy_model.add(tf.keras.layers.Dense(128, activation='tanh'))

# 5. Zmiana architektury
# Przykładowa zmiana architektury - dodanie warstwy Dropout
najlepszy_model.add(tf.keras.layers.Dropout(0.5))

najlepszy_model.fit(x_train, y_train, epochs=5, validation_split=0.1, callbacks=[checkpoint_callback, early_stopping_callback])

# Ponownie ocen najlepszy model
strata, dokladnosc = najlepszy_model.evaluate(x_test, y_test)
print(f"Najlepszy Model po optymalizacji - Strata: {strata}, Dokładność: {dokladnosc}")
najlepszy_model = tf.keras.models.load_model(checkpoint_path)
print(f"Nazwa najlepszego modelu: {najlepszy_model.name}")

print(f"Najlepszy Model przed optymalizacją - Strata: {strata_przed}, Dokładność: {dokladnosc_przed}")

# Wyświetl dokładność przed aktualizacją
print(f"Dokładność przed aktualizacją: {dokladnosc_przed}")

# Wprowadź aktualizacje w modelu

# Ocena modelu po aktualizacji
strata_po, dokladnosc_po = najlepszy_model.evaluate(x_test, y_test)
print(f"Najlepszy Model po optymalizacji - Strata: {strata_po}, Dokładność: {dokladnosc_po}")

# Wyświetl dokładność po aktualizacji
print(f"Dokładność po aktualizacji: {dokladnosc_po}")

# Sprawdź, czy doszło do poprawy
if dokladnosc_po > dokladnosc_przed:
    print("Udana poprawa!")
else:
    print("Brak poprawy lub model się pogorszył.")