import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

# 1. Verileri Yükleme
train = pd.read_csv('customer_churn/train.csv')
test = pd.read_csv('customer_churn/test.csv')
sample_sub = pd.read_csv('customer_churn/sample_submission.csv')

# 2. Veri Ön İşleme
def preprocess(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].median())
    return df

train_clean = preprocess(train)
test_clean = preprocess(test)

features = [c for c in train.columns if c not in ['id', 'Churn']]
cat_features = [c for c in features if train_clean[c].dtype == 'object']

# Kategorik değişkenleri sayısallaştırma
for c in cat_features:
    le = LabelEncoder()
    le.fit(pd.concat([train_clean[c], test_clean[c]]))
    train_clean[c] = le.transform(train_clean[c])
    test_clean[c] = le.transform(test_clean[c])

X_train = train_clean[features]
y_train = (train_clean['Churn'] == 'Yes').astype(int) 
X_test = test_clean[features]

cat_features_indices = [features.index(c) for c in cat_features]

# 3. Model Kurulumu ve Eğitimi
model = HistGradientBoostingClassifier(
    categorical_features=cat_features_indices,
    learning_rate=0.05,
    max_iter=150,
    random_state=42
)

model.fit(X_train, y_train)

# 4. Test Verisi Üzerinde Kesin Sınıf (0 veya 1) Tahmini Yapma
# predict() direkt olarak 0 ya da 1 döndürür
test_preds = model.predict(X_test)

# 5. Submission Dosyasını Hazırlama
sample_sub['Churn'] = test_preds
sample_sub.to_csv('customer_churn/submission.csv', index=False)