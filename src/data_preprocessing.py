import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/raw/petrol_prices_comparison.csv")

# Select useful columns
df = df[['Region','Before_War_USD','Amount_Change','Oil_Import_Dep','Mar7_USD']]

# Handle categorical columns
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("Mar7_USD", axis=1)
y = df["Mar7_USD"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Save processed data
X_train.to_csv("../data/processed/X_train.csv",index=False)
X_test.to_csv("../data/processed/X_test.csv",index=False)
y_train.to_csv("../data/processed/y_train.csv",index=False)
y_test.to_csv("../data/processed/y_test.csv",index=False)

print("Data preprocessing completed")