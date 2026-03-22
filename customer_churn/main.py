import pandas as pd
import numpy as np


train = pd.read_csv("customer_churn/train.csv")
test = pd.read_csv("customer_churn/test.csv")

print(train.corr())