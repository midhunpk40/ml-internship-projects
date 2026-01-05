


# Day 8
# Problem understanding, data loading and basic inspection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Student_Performance.csv")

# Clean column names to avoid errors
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("Dataset Shape:", df.shape)
print("\nColumn Names:\n", df.columns)
print("\nData Types:\n", df.dtypes)
print("\nFirst 5 Rows:\n", df.head())
print("\nLast 5 Rows:\n", df.tail())

# Features and target
X = df.drop("performance_index", axis=1)
y = df["performance_index"]

print("\nFeatures:", X.columns)
print("Target: performance_index")


# Day 9
# Data cleaning: handling missing values and duplicates

print("\nMissing Values Before Cleaning:\n", df.isnull().sum())

# Replace invalid zero values with NaN (if any)
num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].replace(0, np.nan)

# Fill missing values using median
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# Remove duplicate rows
print("\nDuplicates Before Removal:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicates After Removal:", df.duplicated().sum())


# Day 10
# Analysis, scaling and train-test split

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Plot histograms to analyze data distribution
df.hist(figsize=(14, 10), bins=20)
plt.show()

# Display skewness
print("\nSkewness:\n", df.skew(numeric_only=True))

# Identify outliers using boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include="number"))
plt.show()

# Generate correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()

# Separate features and target again
X = df.drop("performance_index", axis=1)
y = df["performance_index"]

# Apply feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# Day 11
# Model training, evaluation and comparison

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\nLinear Regression:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

# KNN Regressor with tuning
print("\nKNN Tuning:")
for k in range(1, 11):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    print(f"K={k}, R2={r2_score(y_test, knn.predict(X_test))}")

# Final KNN model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\nDecision Tree:")
print("MSE:", mean_squared_error(y_test, y_pred_dt))
print("R2 Score:", r2_score(y_test, y_pred_dt))

# Visualize decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, filled=True)
plt.show()

# Model comparison
model_scores = {
    "Linear Regression": r2_score(y_test, y_pred_lr),
    "KNN": r2_score(y_test, y_pred_knn),
    "Decision Tree": r2_score(y_test, y_pred_dt)
}

print("\nModel Comparison:")
for model, score in model_scores.items():
    print(model, ":", score)

print("\nBest Model:", max(model_scores, key=model_scores.get))

print("""
Conclusion:
Student performance prediction was carried out using regression models.
After comparison, the best model was selected based on R2 score.
Proper data cleaning and scaling improved model performance.
""")
