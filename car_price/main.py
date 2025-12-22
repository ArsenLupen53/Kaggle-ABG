import numpy as np
import pandas as pd

veriler = pd.read_csv("car_price/car_price_pred.csv")


veriler = veriler.iloc[:,1:]


first_half = veriler.iloc[:,:7]
second_half = veriler.iloc[:,8:]
car_pice = veriler.iloc[:,7:8].values

car_feature = pd.concat([first_half,second_half],axis=1)
print(car_feature)

#brand -> one-hot encode
#year -> no need
#engine size -> no need
#fuel type -> one hot encode 
#transmission -> binary (label)
#condition -> one hot
#model -> target encoding? label encoding
#scale ederiz random forest modelini uygularız, r2 scoreu falan alrıız