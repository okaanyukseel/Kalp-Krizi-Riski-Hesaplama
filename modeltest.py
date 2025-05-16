import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from imblearn.over_sampling import SMOTE

# Veriyi yükleyelim
df = pd.read_csv('heart.csv')

# Kategorik verileri sayısal değerlere dönüştürelim
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['ChestPainType'] = label_encoder.fit_transform(df['ChestPainType'])
df['RestingECG'] = label_encoder.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = label_encoder.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = label_encoder.fit_transform(df['ST_Slope'])

# Girdi ve çıktıyı belirleyelim
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Veriyi eğitim ve test olarak ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SMOTE ile veri dengesini sağlama
smote = SMOTE(random_state=1)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Veriyi ölçeklendirelim
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# 1. Yapay Sinir Ağları Modeli (YSA)
print('Model 1: Yapay Sinir Ağı')
model = Sequential()
model.add(Dense(16, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Modeli derleyelim
optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Modeli eğitelim
model.fit(X_train_scaled, y_train_smote, epochs=100, verbose=1, validation_data=(X_test_scaled, y_test))

# Test doğruluğu
loss, accuracy = model.evaluate(X_test_scaled, y_test)


# 2. Random Forest Modeli
print('Model 2: Random Forest')
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_smote, y_train_smote)
y_pred_rf = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)


# 3. Karar Ağaçları Modeli
print('Model 3: Karar Ağaçları')
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_smote, y_train_smote)
y_pred_dt = dt_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, y_pred_dt)


# 4. Lojistik Regresyon Modeli
print('Model 4: Lojistik Regresyon')
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_smote, y_train_smote)
y_pred_lr = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, y_pred_lr)






print(f'Yapay Sinir Ağı Test Doğruluk Oranı: {accuracy * 100:.2f}%')
print(f'Random Forest Test Doğruluk Oranı: {rf_accuracy * 100:.2f}%')
print(f'Karar Ağaçları Test Doğruluk Oranı: {dt_accuracy * 100:.2f}%')
print(f'Lojistik Regresyon Test Doğruluk Oranı: {lr_accuracy * 100:.2f}%')
