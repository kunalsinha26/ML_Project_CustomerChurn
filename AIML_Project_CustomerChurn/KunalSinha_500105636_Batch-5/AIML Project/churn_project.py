# churn_project.py

# STEP 1: Problem Definition
"""
Objective: Predict whether a customer will churn (binary classification).
Dataset: Telco Customer Churn Dataset
Success Metric: Accuracy, F1 Score, ROC-AUC
"""

# STEP 2: Data Collection
import pandas as pd

url = url = 'https://raw.githubusercontent.com/sharmaroshan/Telco-Customer-Churn/main/Telco-Customer-Churn.csv'
df = pd.read_csv('Telco-Customer-Churn.csv')

# STEP 3: Data Exploration (EDA)
import seaborn as sns
import matplotlib.pyplot as plt

print("=== Basic Info ===")
print(df.info())

print("\n=== Descriptive Stats ===")
print(df.describe())

print("\n=== Target Class Distribution ===")
print(df['Churn'].value_counts())

sns.countplot(data=df, x='Churn')
plt.title('Class Balance')
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# STEP 4: Data Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Drop customerID
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Encode categorical variables
for col in df.select_dtypes('object').columns:
    if col != 'Churn':
        df[col] = LabelEncoder().fit_transform(df[col])

# Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# STEP 5: Feature Engineering
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=10)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# STEP 6: Model Selection
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

# STEP 7: Model Training
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None]
}

grid = GridSearchCV(model, params, cv=5, scoring='f1')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# STEP 8: Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== ROC-AUC Score ===")
print(roc_auc_score(y_test, y_proba))

# STEP 9: Save Model
import joblib

joblib.dump(best_model, 'best_rf_model.pkl')
print("\nModel saved as best_rf_model.pkl")
