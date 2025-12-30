#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 21:01:08 2025

@author: ahmetberkguneli
"""

#importing libraires
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import statsmodels.api as sm

#preparing datas
train_data = pd.read_csv("pigeon/train_bun.csv")
test_data=pd.read_csv("pigeon/test_bun.csv")

x_train_data = train_data.iloc[:,1:11]
y_train_data = train_data.iloc[:,-1:]
test_data1 = test_data.iloc[:,1:]

#scaling datas for better learning
scaler = StandardScaler()
x_train_data = scaler.fit_transform(x_train_data)
test_data1 = scaler.transform(test_data1)

#splitting train data to see accuracy of models
x_train, x_test, y_train , y_test =train_test_split(x_train_data,y_train_data,test_size=0.2,random_state=42)


#Logistic Regression
model1 = LogisticRegression()
model1.fit(x_train,y_train)

y_pred1 = model1.predict(x_test)
print("Logistic Regression Success Rate:")
print(accuracy_score(y_test,y_pred1)) #0.91
print("Detailed report:")
print(classification_report(y_test,y_pred1))
print("*"*50)

"""
#Checking p-values
x_train = sm.add_constant(x_train)
logit_model = sm.Logit(y_train,x_train)
result = logit_model.fit()

print(result.summary())

print("*"*50)
"""


"""
#Random Forest

from sklearn.ensemble import RandomForestClassifier

rf_cla = RandomForestClassifier(n_estimators = 10,random_state=42)
rf_cla.fit(x_train,y_train)
y_pred2 = rf_cla.predict(x_test)
print(accuracy_score(y_test,y_pred2)) #0.83
"""

#preparing submission
final_pred = model1.predict(test_data1)

data={"id":test_data["id"],"will_eat_bun":final_pred}

submission = pd.DataFrame(data)

submission.to_csv("pigeon/submission.csv",index=False)




