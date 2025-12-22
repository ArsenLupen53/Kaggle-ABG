import pandas as pd
import numpy as np
import matplotlib as plt


#pulling test data
x_test = pd.read_csv("home-data-for-ml-course/test.csv")



#pulling train data
x_data = pd.read_csv("home-data-for-ml-course/train.csv")

x_train = x_data.iloc[:,:-1]
y_train = x_data.iloc[:,-1]