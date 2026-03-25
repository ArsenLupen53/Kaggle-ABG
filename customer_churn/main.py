
#import libraries
import pandas as pd
import numpy as np



#defining datas
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
x_train = train.iloc[:,1:-1]
y_train = train.iloc[:,-1:]
x_test = test.iloc[:,1:]



# Dönüştürmek istediğin sütunların listesi
kategorik_sutunlar = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']

# Train ve Test setlerini tek seferde dönüştür
x_train_encoded = pd.get_dummies(x_train, columns=kategorik_sutunlar, drop_first=True)
x_test_encoded = pd.get_dummies(x_test, columns=kategorik_sutunlar, drop_first=True)


#algoritmaları kullanmak için kütüphane import
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#train test ayırdım
X_train,X_test,y_train,y_test = train_test_split(x_train_encoded,y_train,test_size=0.2,random_state=42)

#Ölçeklendirme
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#modeli tanımla
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

#tahmin yap, değerlendir
y_pred = knn.predict(X_test)
#print(f"KNN Başarı Oranı: {accuracy_score(y_test, y_pred):.2f}")
#0.84 başarı

# test.csv verileri üzerinde tahmin yap (predict)
# x_test_encoded hali hazırda get_dummies ile dönüştürülmüş ve ölçeklendirilmiş olmalıdır
# main.py dosyasındaki scaler objesini kullanarak x_test_encoded'ı ölçeklendiriyoruz
X_test_final = scaler.transform(x_test_encoded)
predictions = knn.predict(X_test_final)

if predictions.dtype == 'object':
    predictions = pd.Series(predictions).map({'No': 0, 'Yes': 1}).values

# sample_submission formatında dataframe oluştur
submission = pd.DataFrame({
    'id': test['id'],
    'Churn': predictions
})

# Churn değerlerini eğer model 0/1 yerine No/Yes döndürüyorsa veya tam tersi ise kontrol et
# submission['Churn'] = submission['Churn'].replace({1: 'Yes', 0: 'No'}) # Gerekirse kullanabilirsin

# Sonucu CSV olarak kaydet
submission.to_csv('customer_churn/submission.csv', index=False)

print("Tahminler başarıyla 'submission.csv' dosyasına kaydedildi.")
