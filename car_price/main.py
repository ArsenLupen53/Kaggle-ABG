
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


veriler = pd.read_csv("car_price/car_price_pred.csv")

print(veriler)

veriler = veriler.iloc[:,1:]


first_half = veriler.iloc[:,:7]
second_half = veriler.iloc[:,8:]
car_price = veriler.iloc[:,7:8].values

car_feature = pd.concat([first_half,second_half],axis=1)


encoder = OneHotEncoder(sparse_output=False)
#brand -> one-hot encode
brand = car_feature.iloc[:,0:1]
brand = encoder.fit_transform(brand)
#year -> no need
#engine size -> no need
year_engine = car_feature.iloc[:,1:3].values
#fuel type -> one hot encode 
fuel_type = car_feature.iloc[:,3:4]
fuel_type = encoder.fit_transform(fuel_type)
#transmission -> binary (label)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
transmission=car_feature.iloc[:,4:5]
transmission = le.fit_transform(transmission)
transmission = transmission.reshape(2500,1)
#mileage
milage = car_feature.iloc[:,5:6].values
#condition -> one hot
condition = car_feature.iloc[:,6:7]
condition = encoder.fit_transform(condition)
#model -> target encoding? label encoding
model = car_feature.iloc[:,7:]
model = le.fit_transform(model)
model = model.reshape(2500,1)


sonuc = np.concatenate((brand,year_engine,fuel_type,milage,condition,transmission,model),axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonuc,car_price,test_size = 0.20, random_state=42)



from sklearn.metrics import r2_score
import statsmodels.api as sm


#scale ederiz random forest modelini uygularız, r2 scoreu falan alrıız

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=100,random_state=42)

rf_reg.fit(x_train,y_train)

prediction = rf_reg.predict(x_test)

print("RANDOM FOREST OLS")
model = sm.OLS(rf_reg.predict(x_test),y_test)
print(model.fit().summary())
print("*"*40)
print("r2 score")
print(r2_score(y_test,rf_reg.predict(x_test)))




