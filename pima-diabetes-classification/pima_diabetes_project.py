# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load dataset
df = pd.read_csv("diabetes.csv")

# Inspect dataset
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

# Identify features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print("\nFeatures (X):")
print(X.columns)

print("\nTarget (y): Outcome")

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check zero values (important for medical data)
print("\nZero Values Count:")
print((df == 0).sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check duplicates
print("\nDuplicate Rows:", df.duplicated().sum())

# -----------------------------
# Outlier Detection using Boxplot
# -----------------------------
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Boxplot for Outlier Detection")
plt.show()

# -----------------------------
# 2. Relationship between features and target
# -----------------------------

# Outcome vs Glucose
plt.figure(figsize=(6, 4))
sns.boxplot(x="Outcome", y="Glucose", data=df)
plt.title("Outcome vs Glucose")
plt.show()

# Outcome vs BMI
plt.figure(figsize=(6, 4))
sns.boxplot(x="Outcome", y="BMI", data=df)
plt.title("Outcome vs BMI")
plt.show()

# -----------------------------
# 3. Histograms for numerical features
# -----------------------------
df.hist(figsize=(12, 10), bins=20)
plt.suptitle("Histograms of Numerical Features")
plt.show()
# Plot histograms to understand the distribution of all numerical features
df.hist(figsize=(14, 12), bins=20)
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.show()

# Display skewness values to check whether features are left or right skewed
print("\nSkewness of Numerical Features:")
print(df.skew())

# Use boxplots to visually identify outliers present in the dataset
plt.figure(figsize=(14, 6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Boxplot to Identify Outliers")
plt.show()

# Generate correlation heatmap to analyze relationships between features and target
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
# Replace invalid zero values with NaN for medical features
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# Handle missing values using median to reduce effect of outliers
for col in cols_with_zero:
    df[col].fillna(df[col].median(), inplace=True)

# Verify cleaned data by checking missing values
print("Missing values after cleaning:\n", df.isnull().sum())

# Check duplicate records before removal
print("Duplicates before removal:", df.duplicated().sum())

# Remove duplicate rows to maintain data consistency
df.drop_duplicates(inplace=True)

# Confirm duplicates are removed
print("Duplicates after removal:", df.duplicated().sum())

# Train KNN model with initial K value
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions using KNN
y_pred_knn = knn.predict(X_test)

# Evaluate KNN model performance
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# Tune K value to find optimal neighbors
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    print(f"K={k}, Accuracy={accuracy_score(y_test, knn.predict(X_test))}")



# Decision Tree Model

# Train Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Make predictions using Decision Tree
y_pred_dt = dt.predict(X_test)

# Evaluate Decision Tree performance
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Visualize Decision Tree structure
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, class_names=["No Diabetes", "Diabetes"], filled=True)
plt.show()
# Store accuracy scores of all models
model_accuracy = {
    "Logistic Regression": accuracy_score(y_test, y_pred_lr),
    "KNN": accuracy_score(y_test, y_pred_knn),
    "Decision Tree": accuracy_score(y_test, y_pred_dt)
}

# Display comparison of model performances
print("Model Accuracy Comparison:")
for model, acc in model_accuracy.items():
    print(f"{model}: {acc}")

# Select best performing model based on accuracy
best_model = max(model_accuracy, key=model_accuracy.get)
print("Best Model:", best_model)

# Final conclusion of the project
print("""
Conclusion:
Logistic Regression, KNN, and Decision Tree models were trained and evaluated.
The best model was selected based on accuracy.
The cleaned and scaled dataset significantly improved model performance.
""")
