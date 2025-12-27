import pandas as pd
import numpy as np
import matplotlib as plt


#pulling test data
x_test = pd.read_csv("house_price/home-data-for-ml-course/test.csv")



#pulling train data
x_data = pd.read_csv("house_price/home-data-for-ml-course/train.csv")

x_train = x_data.iloc[:,:-1]
y_train = x_data.iloc[:,-1]

#train encode etmek lazım
numerical_cols = x_train.select_dtypes(include=["number"]).columns

for col in numerical_cols:
    x_train[col] = x_train[col].fillna(x_train[col].median())

categorical_cols = x_train.select_dtypes(include=["object"]).columns
x_train[categorical_cols] = x_train[categorical_cols].fillna("None")

x_train_ohe = pd.get_dummies(x_train, columns=categorical_cols,drop_first=True)
#x_train_ohe bizim encode tamamlanmış train datasetimiz


#test encode etmek lazım
numerical_cols2 = x_test.select_dtypes(include=["number"]).columns

for col in numerical_cols2:
    x_test[col] = x_test[col].fillna(x_test[col].median())

categorical_cols2 = x_test.select_dtypes(include=["object"]).columns
x_test[categorical_cols2] = x_test[categorical_cols2].fillna("None")

x_test_ohe = pd.get_dummies(x_test,columns=categorical_cols2,drop_first=True)

#train datasetini accuracy görmek için ayırmak lazım
from sklearn.model_selection import train_test_split
x_train_1 , x_test_1 , y_train_1, y_test_1 =train_test_split(x_train_ohe,y_train,test_size=0.20,random_state=42)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import statsmodels.api as sm
rf_reg = RandomForestRegressor(n_estimators=100,random_state=42)
rf_reg.fit(x_train_1,y_train_1)
print("Random Forest OLS")
model5 = sm.OLS(rf_reg.predict(x_test_1),y_test_1)
print(model5.fit().summary())
print("Random Forest R2 değeri:")
print(r2_score(y_test_1,rf_reg.predict(x_test_1)))

x_test_ohe = x_train_ohe.reindex(columns=x_train_ohe.columns,fill_value=0)
test_ids = x_test_ohe["Id"] + 1460
predictions = rf_reg.predict(x_test_ohe)

submission = pd.DataFrame({"Id":test_ids,"SalePrice":predictions})
submission = submission.iloc[:-1]
submission.to_csv("submission2.csv",index=False)









