import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# 1. data load
df = pd.read_csv('./data/diabetes.csv')

# 2. data preprocessing
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3. standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 5. define model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. accuracy check
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. save model
os.makedirs("./src/model", exist_ok=True)
joblib.dump(model, "./src/model/rf_model.pkl")
joblib.dump(scaler, "./src/model/scaler.pkl")

print("✅ saved successfully!")
