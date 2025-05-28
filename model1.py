import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
import matplotlib.pyplot as plt

# Veri yükleme
df = pd.read_csv('heart.csv')

# Kategorik verileri sayısal değerlere dönüştürme
label_encoder = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_cp = LabelEncoder()
df['Sex'] = label_encoder_sex.fit_transform(df['Sex'])
df['ChestPainType'] = label_encoder_cp.fit_transform(df['ChestPainType'])
df['RestingECG'] = label_encoder.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = label_encoder.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = label_encoder.fit_transform(df['ST_Slope'])

# Girdi ve çıktı
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Eğitim ve test verisi olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veri dengesizliğini SMOTE ile giderme
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Model 1: Yapay Sinir Ağı
print('Model 1: Yapay Sinir Ağları')
model = Sequential()
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Modeli eğit, history'yi al
history = model.fit(X_train_scaled, y_train_smote, epochs=50, verbose=1, validation_data=(X_test_scaled, y_test))
loss, accuracy = model.evaluate(X_test_scaled, y_test)
y_pred_nn = (model.predict(X_test_scaled) > 0.5).astype("int32")

# Eğitim ve test doğruluk & kayıp grafiği
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Test Doğruluğu')
plt.title('Model Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Test Kaybı')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.show()

# Model 2: Random Forest
print('Model 2: Random Forest')
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_smote, y_train_smote)
y_pred_rf = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# Model 3: Karar Ağaçları
print('Model 3: Karar Ağaçları')
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_smote, y_train_smote)
y_pred_dt = dt_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

# Model 4: Lojistik Regresyon
print('Model 4: Lojistik Regresyon')
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_smote, y_train_smote)
y_pred_lr = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

# Doğruluk oranları
print(f'Yapay Sinir Ağı Doğruluk Oranı : {accuracy * 100:.2f}%')
print(f'Random Forest Doğruluk Oranı : {rf_accuracy * 100:.2f}%')
print(f'Karar Ağaçları Doğruluk Oranı : {dt_accuracy * 100:.2f}%')
print(f'Lojistik Regresyon Doğruluk Oranı : {lr_accuracy * 100:.2f}%')

# Konfüzyon matrisleri
cm_nn = confusion_matrix(y_test, y_pred_nn)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_lr = confusion_matrix(y_test, y_pred_lr)

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

# ROC eğrileri için tahmin olasılıkları
y_prob_nn = model.predict(X_test_scaled).ravel()
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
y_prob_dt = dt_model.predict_proba(X_test_scaled)[:, 1]
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# ROC değerleri
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_prob_nn)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)

# AUC değerleri
auc_nn = auc(fpr_nn, tpr_nn)
auc_rf = auc(fpr_rf, tpr_rf)
auc_dt = auc(fpr_dt, tpr_dt)
auc_lr = auc(fpr_lr, tpr_lr)

# ROC Eğrisi çizimi
plt.figure(figsize=(10, 8))
plt.plot(fpr_nn, tpr_nn, label=f'Yapay Sinir Ağı (AUC = {auc_nn:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.plot(fpr_dt, tpr_dt, label=f'Karar Ağaçları (AUC = {auc_dt:.2f})')
plt.plot(fpr_lr, tpr_lr, label=f'Lojistik Regresyon (AUC = {auc_lr:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Rasgele Tahmin')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrileri')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
