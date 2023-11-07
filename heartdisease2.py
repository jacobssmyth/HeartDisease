import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = pd.read_csv("/Users/jake/Documents/Coding/Python/Project Stuff/Heart Disease Project/heart_attack_prediction_dataset.csv")

data = data.apply(pd.to_numeric, errors='coerce')
data = data.drop(columns=["Continent", "Country", "Hemisphere"])

X = data.drop(['Heart Attack Risk', 'Patient ID'], axis=1)
y = data['Heart Attack Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int) 

report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{report}')
from tensorflow.keras.models import load_model
model.save('heart_disease_model')
