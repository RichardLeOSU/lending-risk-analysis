import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay)
from sklearn.preprocessing import StandardScaler


# Load cleaned dataset
lending = pd.read_csv("../cleaned_data/lending_club_cleaned.csv")
german = pd.read_csv("../cleaned_data/german_credit_cleaned.csv")


# ----- LINEAR REGRESSION: LENDING CLUB ----

# Keep quantitative columns for Linear Regression
columns_to_keep = ["annual_income", "debt_to_income", "delinq_2y", "inquiries_last_12m",
                   "total_credit_lines", "open_credit_lines", "total_credit_limit",
                   "total_credit_utilized", "loan_amount", "term", "interest_rate",
                   "installment"]

lending_linear = lending[columns_to_keep].copy()

# Make sure all selected columns are numeric
for col in columns_to_keep:
    lending_linear[col] = pd.to_numeric(lending_linear[col], errors="coerce")

# Replace missing values with median
lending_linear = lending_linear.replace([np.inf, -np.inf], np.nan)
lending_linear = lending_linear.fillna(lending_linear.median())

# Final checks
print(lending_linear.head())
print(lending_linear.info())
print(lending_linear.isnull().sum())
print("Duplicate rows:", lending_linear.duplicated().sum())

# Save prepared Linear Regression dataset
lending_linear.to_csv("../cleaned_data/lending_linear_regression_prepared.csv", index=False)

# Set independent variables and dependent variable
x = lending_linear.drop(columns=["installment"])
y = lending_linear["installment"]

# Split data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# Run Linear Regression
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

# Make predictions
y_train_pred = linear_model.predict(x_train)
y_test_pred = linear_model.predict(x_test)

# Get model equation values
intercept = linear_model.intercept_

coefficients = pd.DataFrame({
    "variable": x.columns,
    "coefficient": linear_model.coef_
})

coefficients["coefficient"] = coefficients["coefficient"].round(6)

print("Intercept:", round(intercept, 6))
print(coefficients)

# Save equation values
coefficients.to_csv("../cleaned_data/lending_linear_regression_coefficients.csv", index=False)

# Get model results
training_r2 = r2_score(y_train, y_train_pred)
testing_r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
rmse = mean_squared_error(y_test, y_test_pred) ** 0.5

linear_results = pd.DataFrame({
    "metric": ["training_r2", "testing_r2", "mean_absolute_error", "root_mean_squared_error"],
    "value": [training_r2, testing_r2, mae, rmse]
})

linear_results["value"] = linear_results["value"].round(4)

print(linear_results)

# Save model results
linear_results.to_csv("../cleaned_data/lending_linear_regression_results.csv", index=False)

# Create image of model results
plt.figure(figsize=(8, 2.5))
plt.axis("off")
plt.table(cellText=linear_results.values,
          colLabels=linear_results.columns,
          loc="center")
plt.title("Linear Regression Model Results")
plt.tight_layout()
plt.savefig("../figures/lending_linear_regression_results_table.png", dpi=200)
plt.show()

# Actual vs predicted installment
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)

min_value = min(y_test.min(), y_test_pred.min())
max_value = max(y_test.max(), y_test_pred.max())

plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="red")
plt.title("Actual vs Predicted Installment")
plt.xlabel("Actual Installment")
plt.ylabel("Predicted Installment")
plt.tight_layout()
plt.savefig("../figures/lending_linear_regression_actual_vs_predicted.png", dpi=200)
plt.show()

# Residuals
residuals = y_test - y_test_pred

plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=30, edgecolor="black")
plt.title("Linear Regression Residuals")
plt.xlabel("Actual Installment - Predicted Installment")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("../figures/lending_linear_regression_residuals.png", dpi=200)
plt.show()

# Save predictions
linear_predictions = x_test.copy()
linear_predictions["actual_installment"] = y_test
linear_predictions["predicted_installment"] = y_test_pred
linear_predictions["residual"] = residuals

linear_predictions.to_csv("../cleaned_data/lending_linear_regression_predictions.csv", index=False)

# Example calculation
example = x_test.iloc[[0]].copy()
example_actual = y_test.iloc[0]
example_prediction = linear_model.predict(example)[0]

example_calculation = pd.DataFrame({
    "variable": x.columns,
    "coefficient": linear_model.coef_,
    "example_value": example.iloc[0].values,
    "coefficient_times_value": linear_model.coef_ * example.iloc[0].values
})

intercept_row = pd.DataFrame({
    "variable": ["intercept"],
    "coefficient": [intercept],
    "example_value": [1],
    "coefficient_times_value": [intercept]
})

example_calculation = pd.concat([intercept_row, example_calculation], ignore_index=True)
example_calculation = example_calculation.round(4)

print("Example actual installment:", round(example_actual, 2))
print("Example predicted installment:", round(example_prediction, 2))
print(example_calculation)
print("Manual prediction check:", round(example_calculation["coefficient_times_value"].sum(), 2))

# Save example calculation
example_calculation.to_csv("../cleaned_data/lending_linear_regression_example_calculation.csv", index=False)

# Create image of example calculation
plt.figure(figsize=(12, 5))
plt.axis("off")
plt.table(cellText=example_calculation.values,
          colLabels=example_calculation.columns,
          loc="center")
plt.title("Linear Regression Example Calculation")
plt.tight_layout()
plt.savefig("../figures/lending_linear_regression_example_calculation.png", dpi=200)
plt.show()

# ----- SUPPLEMENTAL LINEAR REGRESSION ANALYSIS -----

# Correlation of each variable with installment
installment_correlations = lending_linear.corr(numeric_only=True)["installment"].drop("installment").sort_values()

print(installment_correlations)

installment_correlations.to_csv("../cleaned_data/lending_installment_correlations.csv")

plt.figure(figsize=(9, 6))
plt.barh(installment_correlations.index, installment_correlations.values)
plt.title("Correlation of Lending Club Variables with Installment")
plt.xlabel("Correlation with Installment")
plt.ylabel("Variable")
plt.tight_layout()
plt.savefig("../figures/lending_installment_correlations.png", dpi=200)
plt.show()


# Residuals vs predicted installment
plt.figure(figsize=(8, 5))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(0, linestyle="--", color="red")
plt.title("Residuals vs Predicted Installment")
plt.xlabel("Predicted Installment")
plt.ylabel("Residual")
plt.tight_layout()
plt.savefig("../figures/lending_linear_regression_residuals_vs_predicted.png", dpi=200)
plt.show()


# Compare full model with a simpler loan-structure model
core_columns = ["loan_amount", "term", "interest_rate"]

x_core = lending_linear[core_columns]
y_core = lending_linear["installment"]

x_core_train, x_core_test, y_core_train, y_core_test = train_test_split(
    x_core, y_core, test_size=0.30, random_state=42
)

core_model = LinearRegression()
core_model.fit(x_core_train, y_core_train)

y_core_test_pred = core_model.predict(x_core_test)

core_results = pd.DataFrame({
    "model": ["full_model", "core_loan_model"],
    "testing_r2": [
        r2_score(y_test, y_test_pred),
        r2_score(y_core_test, y_core_test_pred)
    ],
    "mean_absolute_error": [
        mean_absolute_error(y_test, y_test_pred),
        mean_absolute_error(y_core_test, y_core_test_pred)
    ],
    "root_mean_squared_error": [
        mean_squared_error(y_test, y_test_pred) ** 0.5,
        mean_squared_error(y_core_test, y_core_test_pred) ** 0.5
    ]
})

core_results = core_results.round(4)

print(core_results)

core_results.to_csv("../cleaned_data/lending_linear_regression_model_comparison.csv", index=False)

plt.figure(figsize=(10, 2.5))
plt.axis("off")
plt.table(cellText=core_results.values,
          colLabels=core_results.columns,
          loc="center")
plt.title("Linear Regression Model Comparison")
plt.tight_layout()
plt.savefig("../figures/lending_linear_regression_model_comparison.png", dpi=200)
plt.show()


# ----- LOGISTIC REGRESSION: GERMAN CREDIT -----

# Keep useful columns for Logistic Regression
numeric_columns = ["age", "credit_amount", "duration", "credit_amount_per_month"]
categorical_columns = ["sex", "job", "housing", "saving_accounts", "checking_account"]

columns_to_keep = numeric_columns + categorical_columns + ["credit_risk"]

german_logistic = german[columns_to_keep].copy()

# Create binary label
# credit_risk = 1 stays 0 for lower risk
# credit_risk = 2 becomes 1 for higher risk
german_logistic["bad_credit"] = np.where(german_logistic["credit_risk"] == 2, 1, 0)

# Make sure numeric columns are numeric
for col in numeric_columns:
    german_logistic[col] = pd.to_numeric(german_logistic[col], errors="coerce")

# Replace missing values
german_logistic[numeric_columns] = german_logistic[numeric_columns].replace([np.inf, -np.inf], np.nan)
german_logistic[numeric_columns] = german_logistic[numeric_columns].fillna(german_logistic[numeric_columns].median())

for col in categorical_columns:
    german_logistic[col] = german_logistic[col].fillna("unknown")

# Convert categorical variables into dummy columns
x = german_logistic[numeric_columns + categorical_columns].copy()
x = pd.get_dummies(x, columns=categorical_columns, drop_first=True)

# Convert boolean dummy columns to integers
dummy_columns = x.select_dtypes(include="bool").columns
x[dummy_columns] = x[dummy_columns].astype(int)

y = german_logistic["bad_credit"]

# Final checks
prepared_logistic = x.copy()
prepared_logistic["bad_credit"] = y.values

print(prepared_logistic.head())
print(prepared_logistic.info())
print(prepared_logistic.isnull().sum())
print("Duplicate rows:", prepared_logistic.duplicated().sum())

# Save prepared Logistic Regression dataset
prepared_logistic.to_csv("../cleaned_data/german_logistic_regression_prepared.csv", index=False)


# Class distribution
class_counts = german_logistic["bad_credit"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(["Lower Risk", "Higher Risk"], class_counts.values)
plt.title("German Credit Risk Distribution for Logistic Regression")
plt.xlabel("Credit Risk Group")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../figures/german_logistic_regression_class_distribution.png", dpi=200)
plt.show()


# Split data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=42, stratify=y
)

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# Scale the data
scaler = StandardScaler()

x_train_scaled = pd.DataFrame(
    scaler.fit_transform(x_train),
    columns=x_train.columns,
    index=x_train.index
)

x_test_scaled = pd.DataFrame(
    scaler.transform(x_test),
    columns=x_test.columns,
    index=x_test.index
)

# Run Logistic Regression
logistic_model = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")
logistic_model.fit(x_train_scaled, y_train)

# Make predictions
y_test_pred = logistic_model.predict(x_test_scaled)
y_test_prob = logistic_model.predict_proba(x_test_scaled)[:, 1]

# Save predictions
logistic_predictions = x_test.copy()
logistic_predictions["actual_bad_credit"] = y_test
logistic_predictions["predicted_bad_credit"] = y_test_pred
logistic_predictions["predicted_probability"] = y_test_prob

logistic_predictions.to_csv("../cleaned_data/german_logistic_regression_predictions.csv", index=False)

# Get model equation values
intercept = logistic_model.intercept_[0]

coefficients = pd.DataFrame({
    "variable": x.columns,
    "coefficient": logistic_model.coef_[0],
    "odds_ratio": np.exp(logistic_model.coef_[0])
})

coefficients["coefficient"] = coefficients["coefficient"].round(6)
coefficients["odds_ratio"] = coefficients["odds_ratio"].round(6)

print("Intercept:", round(intercept, 6))
print(coefficients)

# Save coefficients
coefficients.to_csv("../cleaned_data/german_logistic_regression_coefficients.csv", index=False)

# Create image of coefficients
plt.figure(figsize=(12, 4))
plt.axis("off")
plt.table(cellText=coefficients.values,
          colLabels=coefficients.columns,
          loc="center")
plt.title("German Credit Logistic Regression Coefficients and Odds Ratios")
plt.tight_layout()
plt.savefig("../figures/german_logistic_regression_coefficients.png", dpi=200)
plt.show()


# Model results
logistic_results = pd.DataFrame({
    "metric": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
    "value": [
        accuracy_score(y_test, y_test_pred),
        precision_score(y_test, y_test_pred),
        recall_score(y_test, y_test_pred),
        f1_score(y_test, y_test_pred),
        roc_auc_score(y_test, y_test_prob)
    ]
})

logistic_results["value"] = logistic_results["value"].round(4)

print(logistic_results)

# Save model results
logistic_results.to_csv("../cleaned_data/german_logistic_regression_results.csv", index=False)

# Create image of results
plt.figure(figsize=(8, 2.5))
plt.axis("off")
plt.table(cellText=logistic_results.values,
          colLabels=logistic_results.columns,
          loc="center")
plt.title("German Credit Logistic Regression Model Results")
plt.tight_layout()
plt.savefig("../figures/german_logistic_regression_results_table.png", dpi=200)
plt.show()


# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Lower Risk", "Higher Risk"],
            yticklabels=["Lower Risk", "Higher Risk"])
plt.title("German Credit Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("../figures/german_logistic_regression_confusion_matrix.png", dpi=200)
plt.show()


# ROC curve
RocCurveDisplay.from_predictions(y_test, y_test_prob)
plt.title("German Credit Logistic Regression ROC Curve")
plt.tight_layout()
plt.savefig("../figures/german_logistic_regression_roc_curve.png", dpi=200)
plt.show()


# Example calculation
prediction_series = pd.Series(y_test_pred, index=x_test.index)
correct_high_risk = y_test[(y_test == 1) & (prediction_series == 1)].index

if len(correct_high_risk) > 0:
    example_index = correct_high_risk[0]
else:
    example_index = x_test.index[0]

example_original = x_test.loc[[example_index]].copy()
example_scaled = x_test_scaled.loc[[example_index]].copy()
example_actual = y_test.loc[example_index]
example_probability = logistic_model.predict_proba(example_scaled)[0][1]
example_prediction = logistic_model.predict(example_scaled)[0]

example_calculation = pd.DataFrame({
    "variable": x.columns,
    "coefficient": logistic_model.coef_[0],
    "original_value": example_original.iloc[0].values,
    "standardized_value": example_scaled.iloc[0].values,
    "coefficient_times_value": logistic_model.coef_[0] * example_scaled.iloc[0].values
})

intercept_row = pd.DataFrame({
    "variable": ["intercept"],
    "coefficient": [intercept],
    "original_value": [np.nan],
    "standardized_value": [1],
    "coefficient_times_value": [intercept]
})

example_calculation = pd.concat([intercept_row, example_calculation], ignore_index=True)
example_calculation = example_calculation.round(4)

logit_value = example_calculation["coefficient_times_value"].sum()
sigmoid_value = 1 / (1 + np.exp(-logit_value))

example_summary = pd.DataFrame({
    "result": ["logit_value", "sigmoid_probability", "predicted_class", "actual_class"],
    "value": [round(logit_value, 4), round(sigmoid_value, 4), int(example_prediction), int(example_actual)]
})

print("Example probability:", round(example_probability, 4))
print("Example predicted class:", int(example_prediction))
print("Example actual class:", int(example_actual))
print(example_calculation)
print(example_summary)

# Save example calculation
example_calculation.to_csv("../cleaned_data/german_logistic_regression_example_calculation.csv", index=False)
example_summary.to_csv("../cleaned_data/german_logistic_regression_example_summary.csv", index=False)

# Create image of example calculation
plt.figure(figsize=(15, 5))
plt.axis("off")
plt.table(cellText=example_calculation.values,
          colLabels=example_calculation.columns,
          loc="center")
plt.title("German Credit Logistic Regression Example Calculation")
plt.tight_layout()
plt.savefig("../figures/german_logistic_regression_example_calculation.png", dpi=200)
plt.show()

# Create image of sigmoid result
plt.figure(figsize=(8, 2.5))
plt.axis("off")
plt.table(cellText=example_summary.values,
          colLabels=example_summary.columns,
          loc="center")
plt.title("German Credit Logistic Regression Sigmoid Result")
plt.tight_layout()
plt.savefig("../figures/german_logistic_regression_sigmoid_result.png", dpi=200)
plt.show()
