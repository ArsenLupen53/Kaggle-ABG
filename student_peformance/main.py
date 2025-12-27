import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Veriyi Yükle
data = pd.read_csv("student_performance/Student_Performance_Dataset.csv")

# 2. Doğru Sütunları Seç (DÜZELTME BURADA)
# Eski hatalı kod: x_data = data.iloc[:,1:15]
# Yeni kod: 13. (Final_Percentage) ve 14. (Performance_Level) sütunları almiyoruz.
# Sadece Age(1)'den Previous_Year_Score(12)'ye kadar alıyoruz.
x_data = data.iloc[:, 1:13] 
y_data = data.iloc[:, -1]   # Pass_Fail sütunu

# 3. Kategorik Verileri Dönüştür (Encoding)
# x_data için One-Hot Encoding
x_encoded = pd.get_dummies(x_data, drop_first=True)

# y_data (Pass/Fail) için Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y_data)

# 4. Eğitim ve Test Olarak Ayır
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, test_size=0.25, random_state=42)

# 5. Modeli Eğit (Sınıflandırma Modeli Kullanıyoruz)
# Regressor yerine Classifier kullanmalısınız çünkü 0 veya 1 tahmin ediyorsunuz.
rf_model = RandomForestClassifier(n_estimators=10,class_weight="balanced", random_state=42) #class_weight modelin ezberlemesini engeller
rf_model.fit(x_train, y_train)

# 6. Tahmin ve Sonuçları Göster
y_pred = rf_model.predict(x_test)

print("Model Doğruluk Oranı (Accuracy):")
print(accuracy_score(y_test, y_pred))

print("\nDetaylı Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
