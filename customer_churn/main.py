import pandas as pd
import numpy as np


train = pd.read_csv("customer_churn/train.csv")
test = pd.read_csv("customer_churn/test.csv")
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