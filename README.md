# Kalp Hastalığı Tahmin Modeli

Bu proje, bir kişinin kalp hastalığına yakalanma riskini tahmin etmek için çeşitli makine öğrenimi modellerini kullanan bir Python uygulamasıdır. Model, hastaların demografik ve tıbbi verilerini kullanarak tahminler yapar.

## Veri Seti

Veri seti depoda `heart.csv` olarak yer almaktadır (918 kayıt, 12 sütun).

| Sütun | Açıklama (koddaki kullanım) |
|---|---|
| `Age`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak` | Sayısal özellikler |
| `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope` | Kategorik özellikler (`LabelEncoder` ile sayısallaştırılır) |
| `HeartDisease` | Hedef değişken (0 / 1) |

Hedef değişken dağılımı: `0` → 410 kayıt, `1` → 508 kayıt.

## Kullanılan Modeller

Projede dört farklı makine öğrenimi modeli kullanılmıştır:

- **Yapay Sinir Ağları (YSA):** Keras `Sequential` ile kurulmuş, derin öğrenme tabanlı bir model.
- **Random Forest (Rassal Orman):** Bir topluluk öğrenme (ensemble learning) yöntemi.
- **Karar Ağaçları:** Ağaç yapısında bir model.
- **Lojistik Regresyon:** İstatistiksel bir sınıflandırma modeli.

## Yöntem

Her iki model scripti de aynı temel akışı izler:

1. `heart.csv` okunur, kategorik sütunlar `LabelEncoder` ile sayısallaştırılır.
2. Veri `train_test_split` ile %70 eğitim / %30 test olarak ayrılır (`random_state=42`).
3. Eğitim verisine **SMOTE** uygulanarak sınıf dengesi sağlanır.
4. `StandardScaler` ile ölçeklendirme yapılır.
5. Dört model eğitilir ve test doğrulukları konsola yazdırılır.

### `model1.py`

- YSA: `Dense(128, relu)` → `Dense(64, relu)` → `Dense(1, sigmoid)`, Adam (`learning_rate=0.001`), `binary_crossentropy`, 50 epoch.
- SMOTE `random_state=42`, Lojistik Regresyon `max_iter=1000`.
- Görselleştirmeler (matplotlib):
  - YSA eğitim/test doğruluk ve kayıp grafikleri
  - Dört model için karışıklık matrisleri (confusion matrix)
  - Dört model için ROC eğrileri ve AUC değerleri

### `modeltest.py`

- YSA: `Dense(16, relu)` → `Dense(1, sigmoid)`, Adam (`learning_rate=0.001`), 100 epoch.
- SMOTE `random_state=1`.
- Yalnızca dört modelin test doğruluk oranlarını yazdırır (grafik yok).

### `veriyidüzenle.py`

- `heart.csv` dosyasındaki bazı sütunları `LabelEncoder` ile kodlar, sayısal sütunları `StandardScaler` ile ölçekler ve sonucu `heart_düzenlenmis.csv` olarak kaydeder.

## Sonuçlar

Depoda kaydedilmiş bir çıktı bulunmadığından doğruluk/AUC değerleri burada verilmemiştir. Değerler scriptler çalıştırıldığında konsolda ve grafiklerde görüntülenir.

## Proje Yapısı

```
.
├── heart.csv            # Veri seti
├── model1.py            # 4 model + doğruluk, confusion matrix ve ROC grafikleri
├── modeltest.py         # 4 model + test doğruluk oranları
├── veriyidüzenle.py     # Veri ön işleme ve heart_düzenlenmis.csv çıktısı
└── README.md
```

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
python model1.py
# veya
python modeltest.py
```

Scriptler `heart.csv` dosyasını çalışma dizininden okur.

## Gerekli Kütüphaneler

- numpy
- pandas
- scikit-learn
- imbalanced-learn
- tensorflow
- matplotlib
