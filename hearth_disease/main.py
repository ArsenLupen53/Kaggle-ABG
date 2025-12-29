import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#getting datas
data = pd.read_csv("hearth_disease/Heart_Disease_Prediction.csv")

x_data = data.iloc[:,:-1]
y_data = data.iloc[:,-1:]


y_data['Heart Disease'] = y_data['Heart Disease'].map({'Absence': 0, 'Presence': 1})

#Splitting data for trying models
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.20,random_state=42)


#Scaling datas
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)




# 1:LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=42)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)


from sklearn.metrics import f1_score, classification_report, confusion_matrix

f1_skor_lr = f1_score(y_pred,y_test)



# 2:RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=42)
rf_reg.fit(x_train,y_train)
y_pred_rf = rf_reg.predict(x_test)
y_pred_rf = [1 if p >= 0.5 else 0 for p in y_pred_rf]
f1_skor_rf = f1_score(y_pred_rf,y_test)

# 3:XGBOOST
import xgboost as xgb

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')


model.fit(X_train,y_train)

f1_skor_xgb = f1_score(model.predict(X_test),y_test)




print("Logistic Regression f1 score:")
print(f1_skor_lr) #0.8780487804878049
print("*"*50)

print("Random Forest f1 score:")
print(f1_skor_rf) #0.6666666666666666
print("*"*50)

print("XGBoost f1 score:")
print(f1_skor_xgb) #0.75