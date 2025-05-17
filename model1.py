import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

df = pd.read_csv('heart.csv')

# Kategorik verileri sayısal değerlere dönüştürelim
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_cp = LabelEncoder()
df['Sex'] = label_encoder_sex.fit_transform(df['Sex'])
df['ChestPainType'] = label_encoder_cp.fit_transform(df['ChestPainType'])
df['RestingECG'] = label_encoder.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = label_encoder.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = label_encoder.fit_transform(df['ST_Slope'])

#Girdi ve Çıktıyı belirle
X = df.drop('HeartDisease' , axis=1)
y = df['HeartDisease']

#veriyi eğitim ve test olarak ayıralım
X_train, X_test , y_train, y_text = train_test_split(X,y, test_size=0.3 , random_state=42)
#veri dengesizliğini düzelt

smote = SMOTE(random_state=42)
X_train_smote , y_train_smote = smote.fit_resample(X_train, y_train)
#veriyi ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

#1.Yapay sinir ağları
print('Model 1: Yapay Sinir Ağları')
model = Sequential()
model.add(Dense(16, input_dim=X_train_scaled.shape[1] ,activation = 'relu'))
model.add(Dense(16, input_dim=X_train_scaled.shape[1] ,activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
#modeli derleyelim
optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy' , optimizer = optimizer , metrics=['accuracy'])
#modeli eğitielim
model.fit(X_train_scaled, y_train_smote, epochs=50 , verbose=1 , validation_data=(X_test_scaled, y_text) )
loss, accuracy = model.evaluate(X_test_scaled, y_text)

#2. MOdel
print('Model 2 : Random Forest')
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_smote, y_train_smote)
y_pred_rf = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_text, y_pred_rf)
#3. Karar ağaçları Modeli
print('Model 3: Karar Ağaçları')
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_smote , y_train_smote)
y_pred_dt = dt_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_text , y_pred_dt)

#4.lojistijk regresyon modeli
print('Model 4: lojistijk regresyon modeli')
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_smote , y_train_smote)
y_pred_lr = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_text, y_pred_lr)

print(f'Yapay Sinir Ağı Doğruluk Oranı : {accuracy * 100:.2f}%')
print(f'Random Forest Doğruluk Oranı : {rf_accuracy * 100:.2f}%')
print(f'Karar Ağaçları Doğruluk Oranı : {dt_accuracy * 100:.2f}%')
print(f'Lojistik Doğruluk Oranı : {lr_accuracy * 100:.2f}%')


# Kullanıcıdan veri alalım ve tahmin yapalım
while True:
    user_input_1 = float(input('Yaşınızı Giriniz: '))
    user_input_2 = input('Cinsiyetinizi Giriniz (M, F): ')
    user_input_3 = input('Göğüs Ağrısı Tipini Giriniz (ATA, NAP, ASY, TA): ')
    user_input_4 = float(input('Dinlenme Kan Basıncınızı Giriniz: '))
    user_input_5 = float(input('Kolesterol Değerinizi Giriniz: '))

    try:
        # Kullanıcı verilerini sayısal hale getirme
        user_data = pd.DataFrame({
            'Age': [user_input_1],
            'Sex': [label_encoder_sex.transform([user_input_2])[0]],  # Label encoder'ı düzgün kullan
            'ChestPainType': [label_encoder_cp.transform([user_input_3])[0]],  # Aynı şekilde chest pain tipi
            'RestingBP': [user_input_4],
            'Cholesterol': [user_input_5],
            'FastingBS': [0],  # Sabit bir değer
            'RestingECG': [0],  # Sabit bir değer
            'MaxHR': [150],  # Sabit örnek değer
            'ExerciseAngina': [0],  # Sabit bir değer
            'Oldpeak': [0.0],
            'ST_Slope': [1]
        })

        # Veriyi ölçeklendirme
        user_data_scaled = scaler.transform(user_data)

        # Tahmin yapma
        prediction = model.predict(user_data_scaled)
        print(f'Tahmin sonucu: Kalp hastalığı risk oranınız: {prediction[0][0]:.4f}')

    except ValueError as e:
        print(f"Bir hata oluştu: {e}. Lütfen girdiğiniz bilgileri kontrol edin.")

    # Yeni tahmin almak için devam edelim
    devam = input('Başka bir tahmin yapmak istiyor musunuz? (E/H): ')
    if devam.lower() != 'e':
        break