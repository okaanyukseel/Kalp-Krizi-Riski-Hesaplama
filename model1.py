import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
import matplotlib.pyplot as plt

# Veriyi oku
df = pd.read_csv('heart.csv')

# Kategorik verileri sayısal değerlere dönüştürelim
label_encoder = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_cp = LabelEncoder()
df['Sex'] = label_encoder_sex.fit_transform(df['Sex'])
df['ChestPainType'] = label_encoder_cp.fit_transform(df['ChestPainType'])
df['RestingECG'] = label_encoder.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = label_encoder.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = label_encoder.fit_transform(df['ST_Slope'])

# Girdi ve çıktıyı belirle
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Veriyi eğitim ve test olarak ayıralım
X_train, X_test, y_train, y_text = train_test_split(X, y, test_size=0.3, random_state=42)

# Veri dengesizliğini düzelt
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Veriyi ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Model 1: Yapay Sinir Ağları
print('Model 1: Yapay Sinir Ağları')
model = Sequential()
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(X_train_scaled, y_train_smote, epochs=50, verbose=1, validation_data=(X_test_scaled, y_text))
loss, accuracy = model.evaluate(X_test_scaled, y_text)
y_pred_nn = (model.predict(X_test_scaled) > 0.5).astype("int32")

# Model 2: Random Forest
print('Model 2 : Random Forest')
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_smote, y_train_smote)
y_pred_rf = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_text, y_pred_rf)

# Model 3: Karar Ağaçları
print('Model 3: Karar Ağaçları')
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_smote, y_train_smote)
y_pred_dt = dt_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_text, y_pred_dt)

# Model 4: Lojistik Regresyon
print('Model 4: Lojistik Regresyon')
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_smote, y_train_smote)
y_pred_lr = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_text, y_pred_lr)

# Doğruluk oranları
print(f'Yapay Sinir Ağı Doğruluk Oranı : {accuracy * 100:.2f}%')
print(f'Random Forest Doğruluk Oranı : {rf_accuracy * 100:.2f}%')
print(f'Karar Ağaçları Doğruluk Oranı : {dt_accuracy * 100:.2f}%')
print(f'Lojistik Regresyon Doğruluk Oranı : {lr_accuracy * 100:.2f}%')

# Konfüzyon Matrisleri
cm_nn = confusion_matrix(y_text, y_pred_nn)
cm_rf = confusion_matrix(y_text, y_pred_rf)
cm_dt = confusion_matrix(y_text, y_pred_dt)
cm_lr = confusion_matrix(y_text, y_pred_lr)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ConfusionMatrixDisplay(cm_nn, display_labels=['No Disease', 'Disease']).plot(ax=axes[0, 0], cmap='Blues')
axes[0, 0].set_title("Yapay Sinir Ağı")

ConfusionMatrixDisplay(cm_rf, display_labels=['No Disease', 'Disease']).plot(ax=axes[0, 1], cmap='Greens')
axes[0, 1].set_title("Random Forest")

ConfusionMatrixDisplay(cm_dt, display_labels=['No Disease', 'Disease']).plot(ax=axes[1, 0], cmap='Oranges')
axes[1, 0].set_title("Karar Ağaçları")

ConfusionMatrixDisplay(cm_lr, display_labels=['No Disease', 'Disease']).plot(ax=axes[1, 1], cmap='Purples')
axes[1, 1].set_title("Lojistik Regresyon")

plt.tight_layout()
plt.show()
