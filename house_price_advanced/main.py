#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 21:13:49 2025

@author: ahmetberkguneli
"""

import pandas as pd
import numpy as np


#train data çekiyoruz
x_data=pd.read_csv("house_price_advanced/train.csv")
x_train= x_data.iloc[:,:-1]
y_train = x_data.iloc[:,-1]

#test data çekiyoruz
x_test = pd.read_csv("house_price_advanced/test.csv")

#train datasetini encode
numerical_cols = x_train.select_dtypes(include=["number"]).columns

for col in numerical_cols:
    x_train[col] = x_train[col].fillna(x_train[col].median())
    
categorical_cols = x_train.select_dtypes(include=["object"]).columns
x_train[categorical_cols] = x_train[categorical_cols].fillna("None")

x_train_ohe = pd.get_dummies(x_train,columns=categorical_cols,drop_first=True)

#train datasetinin accuracy bakalım
from sklearn.model_selection import train_test_split
x_train_1,x_test_1,y_train_1,y_test_1 = train_test_split(x_train_ohe,y_train,test_size=0.20,random_state=42)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import statsmodels.api as sm

"""
rf_reg=RandomForestRegressor(n_estimators=261,random_state=42)
rf_reg.fit(x_train_1,y_train_1)
print("Random Forest OLS")
model5 = sm.OLS(rf_reg.predict(x_test_1),y_test_1)
print(model5.fit().summary())
print("Random Forest R2 değeri:")
print(r2_score(y_test_1,rf_reg.predict(x_test_1)))"""


from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x_train_1)
sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(y_train_1.values.reshape(-1,1)))
from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)) )
#daha yüksek sonuç aldı


numerical_cols2 = x_test.select_dtypes(include=["number"]).columns

for col in numerical_cols2:
    x_test[col] = x_test[col].fillna(x_test[col].median())

categorical_cols2=x_test.select_dtypes(include=["object"]).columns
x_test[categorical_cols2] = x_test[categorical_cols2].fillna("None")

x_test_ohe = pd.get_dummies(x_test,columns=categorical_cols2,drop_first=True)



#submission ayarlayacağız
sc3 = StandardScaler()
x_test_ohe_2 = x_train_ohe.reindex(columns=x_train_ohe.columns,fill_value=0)
x_test_ohe = sc3.fit_transform(x_test_ohe_2) 
test_ids = x_test_ohe_2["Id"] + 1460
predictions =svr_reg.predict(x_test_ohe)



submission = pd.DataFrame({"Id":test_ids,"SalePrice":predictions})
submission = submission.iloc[:-1]
submission.to_csv("submission.csv",index=False)


