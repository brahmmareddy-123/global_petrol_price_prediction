import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

X_train = pd.read_csv("../data/processed/X_train.csv")
X_test = pd.read_csv("../data/processed/X_test.csv")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open("../artifacts/scaler.pkl","wb") as f:
    pickle.dump(scaler,f)

print("Feature Engineering Completed")