import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Adatok beolvasása
dataset = pd.read_csv('cancer.csv')

# Célváltozó és jellemzők szétválasztása
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]

# Adatfelosztás edző és teszt halmazra
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Modell felépítése
model = tf.keras.models.Sequential()

# Bemeneti réteg, itt az input_shape a jellemzők száma (30)
model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='sigmoid'))

# Rejtett rétegek
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))

# Kimeneti réteg, 1 neuron, mert bináris osztályozás
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Modell fordítása
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modell tanítása
model.fit(x_train, y_train, epochs=1000)

model.evaluate(x_test, y_test)