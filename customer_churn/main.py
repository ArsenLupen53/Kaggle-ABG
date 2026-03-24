#import libraries
import pandas as pd
import numpy as np



#defining datas
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
x_train = train.iloc[:,1:-1]
y_train = train.iloc[:,-1:]
x_test = test.iloc[:,1:]



#splitting datas which will be encoded
train_gender = x_train.iloc[:,0:1]
test_gender = x_test.iloc[:,0:1]

train_partner = x_train.iloc[:,2:3]
test_partner = x_test.iloc[:,2:3]

train_dependents = x_train.iloc[:,3:4]
test_dependents = x_test.iloc[:,3:4]

train_phone = x_train.iloc[:,5:6]
test_phone = x_test.iloc[:,5:6]

train_lines = x_train.iloc[:,6:7]
test_lines = x_test.iloc[:,6:7]

train_internet = x_train.iloc[:,7:8]
test_internet = x_test.iloc[:,7:8]



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

#Destek Vektör Makinesi
from sklearn.svm import SVC

#modeli tanımla
svm_model = SVC(kernel="linear",C=1.0) #C: Hata payı toleransı

#modeli eğit
svm_model.fit(X_train,y_train)

#tahmin yap
y_pred_svm = svm_model.predict(X_test)

#Değerlendir
print(f"SVM Başarı Oranı: {accuracy_score(y_test, y_pred_svm):.2f}")