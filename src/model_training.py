import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X_train = pd.read_csv("../data/processed/X_train.csv")
X_test = pd.read_csv("../data/processed/X_test.csv")
y_train = pd.read_csv("../data/processed/y_train.csv")
y_test = pd.read_csv("../data/processed/y_test.csv")

model = LinearRegression()

model.fit(X_train,y_train)

pred = model.predict(X_test)

score = r2_score(y_test,pred)

print("Model Score:",score)

# Save model
with open("../artifacts/model.pkl","wb") as f:
    pickle.dump(model,f)

print("Model Training Completed")