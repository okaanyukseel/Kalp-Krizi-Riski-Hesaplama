import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


#veriyi yükle

df = pd.read_csv('heart.csv')
# print(df.head())
# print(df.info())
# print(df.isnull().sum())
#kategorik sütunlar

categorical_columns = ['ChestPainType','Sex','RestingBP','ExerciseAngina','ST_Slope']
#Label ile bu sütunları sayısal hale getirelim

le = LabelEncoder()
for column in categorical_columns:
    df[column] = le.fit_transform(df[column])

numeric_columns = ['Age','RestingBP','Cholesterol','FastingBS' , 'MaxHR' , 'Oldpeak']
#verileri standar hale getir
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
print(df.head())


df.to_csv('heart_düzenlenmis.csv' , index=False)




