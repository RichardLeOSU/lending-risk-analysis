import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


# Create output folders if they do not already exist
os.makedirs("../cleaned_data", exist_ok=True)
os.makedirs("../figures", exist_ok=True)


# Load cleaned dataset
german = pd.read_csv("../cleaned_data/german_credit_cleaned.csv")


# ----- SUPPORT VECTOR MACHINE CLASSIFICATION -----


# ----- SVM DATA PREPARATION -----

# Create credit amount per month if it is not already in the cleaned dataset
if "credit_amount_per_month" not in german.columns:
    german["credit_amount_per_month"] = german["credit_amount"] / german["duration"].replace(0, np.nan)

# Keep useful columns for SVM Classification
numeric_columns = ["age", "credit_amount", "duration", "credit_amount_per_month"]
categorical_columns = ["sex", "job", "housing", "saving_accounts", "checking_account", "purpose"]

columns_to_keep = numeric_columns + categorical_columns + ["credit_risk"]

german_svm = german[columns_to_keep].copy()

# Create binary label
# credit_risk = 1 stays 0 for lower risk
# credit_risk = 2 becomes 1 for higher risk
german_svm["bad_credit"] = np.where(german_svm["credit_risk"] == 2, 1, 0)

# Make sure numeric columns are numeric
for col in numeric_columns:
    german_svm[col] = pd.to_numeric(german_svm[col], errors="coerce")

# Replace missing numeric values with median
german_svm[numeric_columns] = german_svm[numeric_columns].replace([np.inf, -np.inf], np.nan)
german_svm[numeric_columns] = german_svm[numeric_columns].fillna(german_svm[numeric_columns].median())

# Replace missing categorical values with unknown
for col in categorical_columns:
    german_svm[col] = german_svm[col].fillna("unknown").astype(str)

# Convert categorical variables into dummy columns
x = german_svm[numeric_columns + categorical_columns].copy()
x = pd.get_dummies(x, columns=categorical_columns, drop_first=True)

# Convert boolean dummy columns to integers
dummy_columns = x.select_dtypes(include="bool").columns
x[dummy_columns] = x[dummy_columns].astype(int)

y = german_svm["bad_credit"]

# Save unscaled prepared SVM dataset
prepared_svm = x.copy()
prepared_svm["bad_credit"] = y.values

print(prepared_svm.head())
print(prepared_svm.info())
print(prepared_svm.isnull().sum())
print("Duplicate rows:", prepared_svm.duplicated().sum())

prepared_svm.to_csv("../cleaned_data/german_svm_prepared.csv", index=False)


# ----- SCALE SVM DATA -----

# SVM models are sensitive to feature scale, so standardize the predictor variables
scaler = StandardScaler()

x_scaled_values = scaler.fit_transform(x)

x_scaled = pd.DataFrame(x_scaled_values, columns=x.columns, index=x.index)

prepared_svm_scaled = x_scaled.copy()
prepared_svm_scaled["bad_credit"] = y.values

print(prepared_svm_scaled.head())
print(prepared_svm_scaled.info())
print(prepared_svm_scaled.isnull().sum())
print("Duplicate rows:", prepared_svm_scaled.duplicated().sum())

prepared_svm_scaled.to_csv("../cleaned_data/german_svm_scaled_prepared.csv", index=False)


# ----- CLASS DISTRIBUTION -----

class_counts = prepared_svm["bad_credit"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(["Lower Risk", "Higher Risk"], class_counts.values)
plt.title("German Credit Risk Distribution for SVM Classification")
plt.xlabel("Credit Risk Group")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../figures/german_svm_class_distribution.png", dpi=200)
plt.show()


# ----- PCA VISUALIZATION OF SVM DATA -----

# Use PCA only for visualization because the full SVM model uses all prepared features
pca = PCA(n_components=2, random_state=42)
x_pca = pca.fit_transform(x_scaled)

svm_pca_preview = pd.DataFrame({
    "PC1": x_pca[:, 0],
    "PC2": x_pca[:, 1],
    "bad_credit": y.values
})

plt.figure(figsize=(8, 6))
plt.scatter(
    svm_pca_preview["PC1"],
    svm_pca_preview["PC2"],
    c=svm_pca_preview["bad_credit"],
    alpha=0.75
)
plt.title("PCA View of Prepared German Credit Data for SVM")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig("../figures/german_svm_pca_data_view.png", dpi=200)
plt.show()

svm_pca_preview.to_csv("../cleaned_data/german_svm_pca_preview.csv", index=False)


# ----- TRAIN TEST SPLIT -----

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.30, random_state=42, stratify=y
)

print("SVM training data shape:", x_train.shape)
print("SVM testing data shape:", x_test.shape)


# ----- SVM KERNEL AND COST COMPARISON -----

# Test multiple kernels and multiple costs to go beyond the minimum requirement
cost_values = [0.1, 1, 10, 100]

svm_kernel_settings = [
    {"kernel_name": "linear", "kernel": "linear", "degree": 3},
    {"kernel_name": "poly_degree_2", "kernel": "poly", "degree": 2},
    {"kernel_name": "poly_degree_3", "kernel": "poly", "degree": 3},
    {"kernel_name": "rbf", "kernel": "rbf", "degree": 3},
    {"kernel_name": "sigmoid", "kernel": "sigmoid", "degree": 3}
]

svm_models = {}
svm_predictions = {}
svm_confusion_matrices = {}
svm_result_rows = []

for setting in svm_kernel_settings:
    for cost in cost_values:

        model_name = setting["kernel_name"] + "_C" + str(cost).replace(".", "_")

        svm_model = SVC(
            C=cost,
            kernel=setting["kernel"],
            degree=setting["degree"],
            gamma="scale",
            class_weight="balanced",
            random_state=42
        )

        svm_model.fit(x_train, y_train)

        predictions = svm_model.predict(x_test)
        cm = confusion_matrix(y_test, predictions)

        svm_models[model_name] = svm_model
        svm_predictions[model_name] = predictions
        svm_confusion_matrices[model_name] = cm

        svm_result_rows.append({
            "model": model_name,
            "kernel_name": setting["kernel_name"],
            "kernel": setting["kernel"],
            "degree": setting["degree"] if setting["kernel"] == "poly" else "not_applicable",
            "cost_c": cost,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
            "support_vectors_total": svm_model.n_support_.sum(),
            "support_vectors_lower_risk": svm_model.n_support_[0],
            "support_vectors_higher_risk": svm_model.n_support_[1],
            "true_lower_risk": cm[0, 0],
            "lower_risk_predicted_higher_risk": cm[0, 1],
            "higher_risk_predicted_lower_risk": cm[1, 0],
            "true_higher_risk": cm[1, 1]
        })

svm_results = pd.DataFrame(svm_result_rows)
svm_results = svm_results.sort_values(["f1_score", "accuracy"], ascending=False)
svm_results = svm_results.round(4)

print(svm_results)

svm_results.to_csv("../cleaned_data/german_svm_kernel_cost_results.csv", index=False)


# ----- SVM MODEL RESULTS TABLE IMAGE -----

svm_results_preview = svm_results[[
    "model", "kernel", "degree", "cost_c", "accuracy", "precision",
    "recall", "f1_score", "support_vectors_total"
]].head(12).copy()

plt.figure(figsize=(16, 4))
plt.axis("off")
table = plt.table(cellText=svm_results_preview.values,
                  colLabels=svm_results_preview.columns,
                  loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.4)
plt.title("Top German Credit SVM Kernel and Cost Results")
plt.tight_layout()
plt.savefig("../figures/german_svm_top_kernel_cost_results.png", dpi=200, bbox_inches="tight")
plt.show()


# ----- BEST MODEL FROM EACH KERNEL FAMILY -----

best_svm_by_kernel = (
    svm_results.sort_values(["kernel_name", "f1_score", "accuracy"], ascending=[True, False, False])
    .groupby("kernel_name")
    .head(1)
    .reset_index(drop=True)
)

print(best_svm_by_kernel)

best_svm_by_kernel.to_csv("../cleaned_data/german_svm_best_model_by_kernel.csv", index=False)

best_svm_by_kernel_preview = best_svm_by_kernel[[
    "model", "kernel", "degree", "cost_c", "accuracy", "precision",
    "recall", "f1_score", "support_vectors_total"
]].copy()

plt.figure(figsize=(15, 3.5))
plt.axis("off")
table = plt.table(cellText=best_svm_by_kernel_preview.values,
                  colLabels=best_svm_by_kernel_preview.columns,
                  loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.4)
plt.title("Best SVM Model From Each Kernel Family")
plt.tight_layout()
plt.savefig("../figures/german_svm_best_model_by_kernel.png", dpi=200, bbox_inches="tight")
plt.show()


# ----- COST COMPARISON PLOT -----

plt.figure(figsize=(10, 6))

for kernel_name in svm_results["kernel_name"].unique():
    kernel_subset = svm_results[svm_results["kernel_name"] == kernel_name].sort_values("cost_c")
    plt.plot(kernel_subset["cost_c"], kernel_subset["accuracy"], marker="o", label=kernel_name)

plt.xscale("log")
plt.title("SVM Accuracy by Kernel and Cost")
plt.xlabel("Cost C")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/german_svm_accuracy_by_kernel_and_cost.png", dpi=200)
plt.show()


# ----- F1 SCORE COMPARISON PLOT -----

plt.figure(figsize=(10, 6))

for kernel_name in svm_results["kernel_name"].unique():
    kernel_subset = svm_results[svm_results["kernel_name"] == kernel_name].sort_values("cost_c")
    plt.plot(kernel_subset["cost_c"], kernel_subset["f1_score"], marker="o", label=kernel_name)

plt.xscale("log")
plt.title("SVM F1 Score by Kernel and Cost")
plt.xlabel("Cost C")
plt.ylabel("F1 Score")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/german_svm_f1_by_kernel_and_cost.png", dpi=200)
plt.show()


# ----- CONFUSION MATRICES FOR BEST MODEL FROM EACH KERNEL FAMILY -----

svm_confusion_summary_rows = []

for i in range(len(best_svm_by_kernel)):

    model_name = best_svm_by_kernel.loc[i, "model"]
    kernel_name = best_svm_by_kernel.loc[i, "kernel_name"]
    cost = best_svm_by_kernel.loc[i, "cost_c"]
    cm = svm_confusion_matrices[model_name]

    svm_confusion_summary_rows.append({
        "model": model_name,
        "kernel_name": kernel_name,
        "cost_c": cost,
        "true_lower_risk": cm[0, 0],
        "lower_risk_predicted_higher_risk": cm[0, 1],
        "higher_risk_predicted_lower_risk": cm[1, 0],
        "true_higher_risk": cm[1, 1],
        "accuracy": accuracy_score(y_test, svm_predictions[model_name]),
        "precision": precision_score(y_test, svm_predictions[model_name], zero_division=0),
        "recall": recall_score(y_test, svm_predictions[model_name], zero_division=0),
        "f1_score": f1_score(y_test, svm_predictions[model_name], zero_division=0)
    })

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Lower Risk", "Higher Risk"],
                yticklabels=["Lower Risk", "Higher Risk"])
    plt.title("SVM Confusion Matrix: " + kernel_name + " Kernel, C=" + str(cost))
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig("../figures/german_svm_" + kernel_name + "_best_confusion_matrix.png", dpi=200)
    plt.show()

svm_confusion_summary = pd.DataFrame(svm_confusion_summary_rows)
svm_confusion_summary = svm_confusion_summary.round(4)

print(svm_confusion_summary)

svm_confusion_summary.to_csv("../cleaned_data/german_svm_confusion_summary.csv", index=False)

plt.figure(figsize=(15, 3.5))
plt.axis("off")
table = plt.table(cellText=svm_confusion_summary.values,
                  colLabels=svm_confusion_summary.columns,
                  loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.4)
plt.title("SVM Confusion Matrix Summary by Best Kernel Model")
plt.tight_layout()
plt.savefig("../figures/german_svm_confusion_summary.png", dpi=200, bbox_inches="tight")
plt.show()


# ----- SUPPORT VECTOR SUMMARY -----

support_vector_summary = best_svm_by_kernel[[
    "model",
    "kernel_name",
    "cost_c",
    "support_vectors_total",
    "support_vectors_lower_risk",
    "support_vectors_higher_risk"
]].copy()

print(support_vector_summary)

support_vector_summary.to_csv("../cleaned_data/german_svm_support_vector_summary.csv", index=False)

plt.figure(figsize=(10, 5))
plt.bar(support_vector_summary["kernel_name"], support_vector_summary["support_vectors_total"])
plt.title("Support Vector Count by Best SVM Kernel Model")
plt.xlabel("Kernel")
plt.ylabel("Number of Support Vectors")
plt.tight_layout()
plt.savefig("../figures/german_svm_support_vector_count.png", dpi=200)
plt.show()


# ----- LINEAR SVM FEATURE IMPORTANCE USING COEFFICIENTS -----

# Use the best linear SVM model to examine which scaled features had the largest coefficients
linear_best_row = best_svm_by_kernel[best_svm_by_kernel["kernel_name"] == "linear"].iloc[0]
linear_best_model_name = linear_best_row["model"]
linear_best_model = svm_models[linear_best_model_name]

linear_coefficients = pd.DataFrame({
    "feature": x_scaled.columns,
    "coefficient": linear_best_model.coef_[0]
})

linear_coefficients["absolute_coefficient"] = linear_coefficients["coefficient"].abs()
linear_coefficients = linear_coefficients.sort_values("absolute_coefficient", ascending=False)
linear_coefficients["coefficient"] = linear_coefficients["coefficient"].round(4)
linear_coefficients["absolute_coefficient"] = linear_coefficients["absolute_coefficient"].round(4)

print(linear_coefficients.head(12))

linear_coefficients.to_csv("../cleaned_data/german_svm_linear_feature_coefficients.csv", index=False)

linear_coefficients_preview = linear_coefficients.head(12).copy()

plt.figure(figsize=(11, 4))
plt.axis("off")
table = plt.table(cellText=linear_coefficients_preview.values,
                  colLabels=linear_coefficients_preview.columns,
                  loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.4)
plt.title("Top Linear SVM Feature Coefficients")
plt.tight_layout()
plt.savefig("../figures/german_svm_linear_feature_coefficients_table.png", dpi=200, bbox_inches="tight")
plt.show()

top_linear_features = linear_coefficients.head(12).sort_values("absolute_coefficient")

plt.figure(figsize=(9, 6))
plt.barh(top_linear_features["feature"], top_linear_features["absolute_coefficient"])
plt.title("Top Linear SVM Feature Coefficients")
plt.xlabel("Absolute Coefficient")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("../figures/german_svm_linear_feature_coefficients.png", dpi=200)
plt.show()


# ----- PCA DECISION BOUNDARY VISUALIZATIONS -----

# These are extra visualizations.
# They train separate SVM models on the first two PCA components so the boundary can be shown in 2D.
x_pca_train, x_pca_test, y_pca_train, y_pca_test = train_test_split(
    x_pca, y, test_size=0.30, random_state=42, stratify=y
)

pca_boundary_rows = []

for i in range(len(best_svm_by_kernel)):

    kernel_name = best_svm_by_kernel.loc[i, "kernel_name"]
    kernel = best_svm_by_kernel.loc[i, "kernel"]
    cost = best_svm_by_kernel.loc[i, "cost_c"]
    degree_value = best_svm_by_kernel.loc[i, "degree"]

    if degree_value == "not_applicable":
        degree_value = 3

    pca_svm_model = SVC(
        C=cost,
        kernel=kernel,
        degree=int(degree_value),
        gamma="scale",
        class_weight="balanced",
        random_state=42
    )

    pca_svm_model.fit(x_pca_train, y_pca_train)

    pca_predictions = pca_svm_model.predict(x_pca_test)

    pca_accuracy = accuracy_score(y_pca_test, pca_predictions)
    pca_f1 = f1_score(y_pca_test, pca_predictions, zero_division=0)

    pca_boundary_rows.append({
        "kernel_name": kernel_name,
        "kernel": kernel,
        "cost_c": cost,
        "degree": degree_value,
        "pca_accuracy": pca_accuracy,
        "pca_f1_score": pca_f1
    })

    x_min, x_max = x_pca[:, 0].min() - 1, x_pca[:, 0].max() + 1
    y_min, y_max = x_pca[:, 1].min() - 1, x_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid_predictions = pca_svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    grid_predictions = grid_predictions.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, grid_predictions, alpha=0.25)
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, alpha=0.75, edgecolor="k", linewidth=0.2)
    plt.title("PCA SVM Decision Boundary: " + kernel_name + " Kernel, C=" + str(cost))
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig("../figures/german_svm_pca_boundary_" + kernel_name + ".png", dpi=200)
    plt.show()

pca_boundary_results = pd.DataFrame(pca_boundary_rows)
pca_boundary_results = pca_boundary_results.round(4)

print(pca_boundary_results)

pca_boundary_results.to_csv("../cleaned_data/german_svm_pca_boundary_results.csv", index=False)


# ----- FINAL SVM SUMMARY -----

final_svm_summary = pd.DataFrame({
    "item": [
        "records_used",
        "features_before_scaling",
        "training_records",
        "testing_records",
        "kernels_tested",
        "cost_values_tested",
        "best_model_by_f1",
        "best_model_accuracy",
        "best_model_precision",
        "best_model_recall",
        "best_model_f1_score"
    ],
    "value": [
        prepared_svm.shape[0],
        x.shape[1],
        x_train.shape[0],
        x_test.shape[0],
        ", ".join(best_svm_by_kernel["kernel_name"].tolist()),
        ", ".join([str(c) for c in cost_values]),
        svm_results.iloc[0]["model"],
        svm_results.iloc[0]["accuracy"],
        svm_results.iloc[0]["precision"],
        svm_results.iloc[0]["recall"],
        svm_results.iloc[0]["f1_score"]
    ]
})

print(final_svm_summary)

final_svm_summary.to_csv("../cleaned_data/german_svm_final_summary.csv", index=False)

plt.figure(figsize=(12, 4))
plt.axis("off")
table = plt.table(cellText=final_svm_summary.values,
                  colLabels=final_svm_summary.columns,
                  loc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.4)
plt.title("Final SVM Classification Summary")
plt.tight_layout()
plt.savefig("../figures/german_svm_final_summary.png", dpi=200, bbox_inches="tight")
plt.show()


# ----- ENSEMBLE CLASSIFICATION -----


# ----- ENSEMBLE DATA PREPARATION -----

# Ensemble models will use the same prepared German Credit data as the SVM section.
# The target is bad_credit:
# 0 = Lower Risk
# 1 = Higher Risk

x_ensemble = x.copy()
y_ensemble = y.copy()

prepared_ensemble = x_ensemble.copy()
prepared_ensemble["bad_credit"] = y_ensemble.values

print(prepared_ensemble.head())
print(prepared_ensemble.info())
print(prepared_ensemble.isnull().sum())
print("Duplicate rows:", prepared_ensemble.duplicated().sum())

prepared_ensemble.to_csv("../cleaned_data/german_ensemble_prepared.csv", index=False)

# Use the  70/30 split structure for ensemble models
x_ensemble_train, x_ensemble_test, y_ensemble_train, y_ensemble_test = train_test_split(
    x_ensemble, y_ensemble, test_size=0.30, random_state=42, stratify=y_ensemble
)

# Keep a scaled version for Stacking models that include Logistic Regression or SVM
x_ensemble_scaled = x_scaled.copy()

x_ensemble_scaled_train, x_ensemble_scaled_test, y_ensemble_scaled_train, y_ensemble_scaled_test = train_test_split(
    x_ensemble_scaled, y_ensemble, test_size=0.30, random_state=42, stratify=y_ensemble
)

print("Ensemble training data shape:", x_ensemble_train.shape)
print("Ensemble testing data shape:", x_ensemble_test.shape)


# Create image of class distribution for ensemble models
ensemble_class_counts = prepared_ensemble["bad_credit"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(["Lower Risk", "Higher Risk"], ensemble_class_counts.values)
plt.title("German Credit Risk Distribution for Ensemble Classification")
plt.xlabel("Credit Risk Group")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../figures/german_ensemble_class_distribution.png", dpi=200)
plt.show()


# ----- RANDOM FOREST CLASSIFICATION -----

# Test multiple Random Forest settings to go beyond the minimum requirement
random_forest_settings = []

for n_trees in [100, 200, 500]:
    for max_depth in [3, 5, 8, None]:
        for min_leaf in [5, 10, 15]:
            random_forest_settings.append({
                "n_estimators": n_trees,
                "max_depth": max_depth,
                "min_samples_leaf": min_leaf
            })

random_forest_models = {}
random_forest_predictions = {}
random_forest_confusion_matrices = {}
random_forest_result_rows = []

for setting in random_forest_settings:

    model_name = (
        "rf_"
        + str(setting["n_estimators"])
        + "_trees_depth_"
        + str(setting["max_depth"])
        + "_leaf_"
        + str(setting["min_samples_leaf"])
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=setting["n_estimators"],
        max_depth=setting["max_depth"],
        min_samples_leaf=setting["min_samples_leaf"],
        class_weight="balanced",
        random_state=42
    )

    random_forest_model.fit(x_ensemble_train, y_ensemble_train)

    predictions = random_forest_model.predict(x_ensemble_test)
    cm = confusion_matrix(y_ensemble_test, predictions)

    random_forest_models[model_name] = random_forest_model
    random_forest_predictions[model_name] = predictions
    random_forest_confusion_matrices[model_name] = cm

    random_forest_result_rows.append({
        "model": model_name,
        "n_estimators": setting["n_estimators"],
        "max_depth": setting["max_depth"],
        "min_samples_leaf": setting["min_samples_leaf"],
        "accuracy": accuracy_score(y_ensemble_test, predictions),
        "precision": precision_score(y_ensemble_test, predictions, zero_division=0),
        "recall": recall_score(y_ensemble_test, predictions, zero_division=0),
        "f1_score": f1_score(y_ensemble_test, predictions, zero_division=0),
        "true_lower_risk": cm[0, 0],
        "lower_risk_predicted_higher_risk": cm[0, 1],
        "higher_risk_predicted_lower_risk": cm[1, 0],
        "true_higher_risk": cm[1, 1]
    })

random_forest_results = pd.DataFrame(random_forest_result_rows)
random_forest_results = random_forest_results.sort_values(["f1_score", "accuracy"], ascending=False)
random_forest_results = random_forest_results.round(4)

print(random_forest_results)

random_forest_results.to_csv("../cleaned_data/german_random_forest_results.csv", index=False)


# Create image of top Random Forest results
random_forest_results_preview = random_forest_results[[
    "model", "n_estimators", "max_depth", "min_samples_leaf",
    "accuracy", "precision", "recall", "f1_score"
]].head(12).copy()

plt.figure(figsize=(16, 4))
plt.axis("off")
table = plt.table(cellText=random_forest_results_preview.values,
                  colLabels=random_forest_results_preview.columns,
                  loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.4)
plt.title("Top German Credit Random Forest Results")
plt.tight_layout()
plt.savefig("../figures/german_random_forest_top_results.png", dpi=200, bbox_inches="tight")
plt.show()


# Best Random Forest model
best_random_forest_name = random_forest_results.iloc[0]["model"]
best_random_forest_model = random_forest_models[best_random_forest_name]
best_random_forest_predictions = random_forest_predictions[best_random_forest_name]
best_random_forest_cm = random_forest_confusion_matrices[best_random_forest_name]

print("Best Random Forest model:", best_random_forest_name)
print(best_random_forest_cm)

# Confusion matrix for best Random Forest
plt.figure(figsize=(6, 5))
sns.heatmap(best_random_forest_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Lower Risk", "Higher Risk"],
            yticklabels=["Lower Risk", "Higher Risk"])
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("../figures/german_random_forest_confusion_matrix.png", dpi=200)
plt.show()


# Random Forest feature importances
random_forest_feature_importance = pd.DataFrame({
    "feature": x_ensemble.columns,
    "importance": best_random_forest_model.feature_importances_
})

random_forest_feature_importance = random_forest_feature_importance.sort_values("importance", ascending=False)
random_forest_feature_importance["importance"] = random_forest_feature_importance["importance"].round(4)

print(random_forest_feature_importance.head(12))

random_forest_feature_importance.to_csv("../cleaned_data/german_random_forest_feature_importance.csv", index=False)

random_forest_feature_importance_preview = random_forest_feature_importance.head(12).copy()

plt.figure(figsize=(11, 4))
plt.axis("off")
table = plt.table(cellText=random_forest_feature_importance_preview.values,
                  colLabels=random_forest_feature_importance_preview.columns,
                  loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.4)
plt.title("Top Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("../figures/german_random_forest_feature_importance_table.png", dpi=200, bbox_inches="tight")
plt.show()

top_random_forest_features = random_forest_feature_importance.head(12).sort_values("importance")

plt.figure(figsize=(9, 6))
plt.barh(top_random_forest_features["feature"], top_random_forest_features["importance"])
plt.title("Top Random Forest Feature Importances")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("../figures/german_random_forest_feature_importance.png", dpi=200)
plt.show()


# ----- RANDOM FOREST THREE-TREE VISUALIZATION -----

# Create a smaller Random Forest with 3 trees for visual illustration
best_rf_depth = random_forest_results.iloc[0]["max_depth"]
best_rf_leaf = int(random_forest_results.iloc[0]["min_samples_leaf"])

if best_rf_depth == "None" or pd.isna(best_rf_depth):
    best_rf_depth = None
else:
    best_rf_depth = int(best_rf_depth)

random_forest_three_trees = RandomForestClassifier(
    n_estimators=3,
    max_depth=best_rf_depth,
    min_samples_leaf=best_rf_leaf,
    class_weight="balanced",
    random_state=42
)

random_forest_three_trees.fit(x_ensemble_train, y_ensemble_train)

for tree_number in range(3):

    plt.figure(figsize=(22, 10))
    plot_tree(
        random_forest_three_trees.estimators_[tree_number],
        feature_names=x_ensemble.columns,
        class_names=["Lower Risk", "Higher Risk"],
        rounded=True,
        fontsize=7
    )
    plt.title("Random Forest Example Tree " + str(tree_number + 1))
    plt.tight_layout()
    plt.savefig("../figures/german_random_forest_example_tree_" + str(tree_number + 1) + ".png", dpi=200)
    plt.show()


# ----- ADABOOST CLASSIFICATION -----

# Test multiple AdaBoost settings
adaboost_settings = []

for n_estimators in [50, 100, 200, 500]:
    for learning_rate in [0.01, 0.1, 0.5, 1.0]:
        for max_depth in [1, 2, 3]:
            adaboost_settings.append({
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth
            })

adaboost_models = {}
adaboost_predictions = {}
adaboost_confusion_matrices = {}
adaboost_result_rows = []

for setting in adaboost_settings:

    model_name = (
        "ada_"
        + str(setting["n_estimators"])
        + "_estimators_lr_"
        + str(setting["learning_rate"]).replace(".", "_")
        + "_depth_"
        + str(setting["max_depth"])
    )

    adaboost_base_tree = DecisionTreeClassifier(
        max_depth=setting["max_depth"],
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42
    )

    try:
        adaboost_model = AdaBoostClassifier(
            estimator=adaboost_base_tree,
            n_estimators=setting["n_estimators"],
            learning_rate=setting["learning_rate"],
            random_state=42
        )
    except TypeError:
        adaboost_model = AdaBoostClassifier(
            base_estimator=adaboost_base_tree,
            n_estimators=setting["n_estimators"],
            learning_rate=setting["learning_rate"],
            random_state=42
        )

    adaboost_model.fit(x_ensemble_train, y_ensemble_train)

    predictions = adaboost_model.predict(x_ensemble_test)
    cm = confusion_matrix(y_ensemble_test, predictions)

    adaboost_models[model_name] = adaboost_model
    adaboost_predictions[model_name] = predictions
    adaboost_confusion_matrices[model_name] = cm

    adaboost_result_rows.append({
        "model": model_name,
        "n_estimators": setting["n_estimators"],
        "learning_rate": setting["learning_rate"],
        "base_tree_max_depth": setting["max_depth"],
        "accuracy": accuracy_score(y_ensemble_test, predictions),
        "precision": precision_score(y_ensemble_test, predictions, zero_division=0),
        "recall": recall_score(y_ensemble_test, predictions, zero_division=0),
        "f1_score": f1_score(y_ensemble_test, predictions, zero_division=0),
        "true_lower_risk": cm[0, 0],
        "lower_risk_predicted_higher_risk": cm[0, 1],
        "higher_risk_predicted_lower_risk": cm[1, 0],
        "true_higher_risk": cm[1, 1]
    })

adaboost_results = pd.DataFrame(adaboost_result_rows)
adaboost_results = adaboost_results.sort_values(["f1_score", "accuracy"], ascending=False)
adaboost_results = adaboost_results.round(4)

print(adaboost_results)

adaboost_results.to_csv("../cleaned_data/german_adaboost_results.csv", index=False)


# Create image of top AdaBoost results
adaboost_results_preview = adaboost_results[[
    "model", "n_estimators", "learning_rate", "base_tree_max_depth",
    "accuracy", "precision", "recall", "f1_score"
]].head(12).copy()

plt.figure(figsize=(16, 4))
plt.axis("off")
table = plt.table(cellText=adaboost_results_preview.values,
                  colLabels=adaboost_results_preview.columns,
                  loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.4)
plt.title("Top German Credit AdaBoost Results")
plt.tight_layout()
plt.savefig("../figures/german_adaboost_top_results.png", dpi=200, bbox_inches="tight")
plt.show()


# Best AdaBoost model
best_adaboost_name = adaboost_results.iloc[0]["model"]
best_adaboost_model = adaboost_models[best_adaboost_name]
best_adaboost_predictions = adaboost_predictions[best_adaboost_name]
best_adaboost_cm = adaboost_confusion_matrices[best_adaboost_name]

print("Best AdaBoost model:", best_adaboost_name)
print(best_adaboost_cm)

# Confusion matrix for best AdaBoost
plt.figure(figsize=(6, 5))
sns.heatmap(best_adaboost_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Lower Risk", "Higher Risk"],
            yticklabels=["Lower Risk", "Higher Risk"])
plt.title("AdaBoost Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("../figures/german_adaboost_confusion_matrix.png", dpi=200)
plt.show()


# AdaBoost feature importances
adaboost_feature_importance = pd.DataFrame({
    "feature": x_ensemble.columns,
    "importance": best_adaboost_model.feature_importances_
})

adaboost_feature_importance = adaboost_feature_importance.sort_values("importance", ascending=False)
adaboost_feature_importance["importance"] = adaboost_feature_importance["importance"].round(4)

print(adaboost_feature_importance.head(12))

adaboost_feature_importance.to_csv("../cleaned_data/german_adaboost_feature_importance.csv", index=False)

adaboost_feature_importance_preview = adaboost_feature_importance.head(12).copy()

plt.figure(figsize=(11, 4))
plt.axis("off")
table = plt.table(cellText=adaboost_feature_importance_preview.values,
                  colLabels=adaboost_feature_importance_preview.columns,
                  loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.4)
plt.title("Top AdaBoost Feature Importances")
plt.tight_layout()
plt.savefig("../figures/german_adaboost_feature_importance_table.png", dpi=200, bbox_inches="tight")
plt.show()

top_adaboost_features = adaboost_feature_importance.head(12).sort_values("importance")

plt.figure(figsize=(9, 6))
plt.barh(top_adaboost_features["feature"], top_adaboost_features["importance"])
plt.title("Top AdaBoost Feature Importances")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("../figures/german_adaboost_feature_importance.png", dpi=200)
plt.show()


# ----- STACKING CLASSIFICATION -----

# Stacking combines multiple different models and lets a final model learn from their predictions.

stacking_settings = [
    {
        "model_name": "stack_logistic_final",
        "final_estimator": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    },
    {
        "model_name": "stack_random_forest_final",
        "final_estimator": RandomForestClassifier(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42
        )
    }
]

stacking_models = {}
stacking_predictions = {}
stacking_confusion_matrices = {}
stacking_result_rows = []

for setting in stacking_settings:

    base_estimators = [
        ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ("decision_tree", DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, class_weight="balanced", random_state=42)),
        ("random_forest", RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=42)),
        ("gaussian_nb", GaussianNB()),
        ("svm_rbf", SVC(C=1, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=42))
    ]

    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=setting["final_estimator"],
        stack_method="auto",
        cv=5,
        n_jobs=-1
    )

    stacking_model.fit(x_ensemble_scaled_train, y_ensemble_scaled_train)

    predictions = stacking_model.predict(x_ensemble_scaled_test)
    cm = confusion_matrix(y_ensemble_scaled_test, predictions)

    stacking_models[setting["model_name"]] = stacking_model
    stacking_predictions[setting["model_name"]] = predictions
    stacking_confusion_matrices[setting["model_name"]] = cm

    stacking_result_rows.append({
        "model": setting["model_name"],
        "base_models": "logistic, decision_tree, random_forest, gaussian_nb, svm_rbf",
        "final_estimator": setting["final_estimator"].__class__.__name__,
        "accuracy": accuracy_score(y_ensemble_scaled_test, predictions),
        "precision": precision_score(y_ensemble_scaled_test, predictions, zero_division=0),
        "recall": recall_score(y_ensemble_scaled_test, predictions, zero_division=0),
        "f1_score": f1_score(y_ensemble_scaled_test, predictions, zero_division=0),
        "true_lower_risk": cm[0, 0],
        "lower_risk_predicted_higher_risk": cm[0, 1],
        "higher_risk_predicted_lower_risk": cm[1, 0],
        "true_higher_risk": cm[1, 1]
    })

stacking_results = pd.DataFrame(stacking_result_rows)
stacking_results = stacking_results.sort_values(["f1_score", "accuracy"], ascending=False)
stacking_results = stacking_results.round(4)

print(stacking_results)

stacking_results.to_csv("../cleaned_data/german_stacking_results.csv", index=False)


# Create image of Stacking results
stacking_results_preview = stacking_results[[
    "model", "final_estimator", "accuracy", "precision", "recall", "f1_score"
]].copy()

plt.figure(figsize=(12, 3))
plt.axis("off")
table = plt.table(cellText=stacking_results_preview.values,
                  colLabels=stacking_results_preview.columns,
                  loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.4)
plt.title("German Credit Stacking Results")
plt.tight_layout()
plt.savefig("../figures/german_stacking_results.png", dpi=200, bbox_inches="tight")
plt.show()


# Best Stacking model
best_stacking_name = stacking_results.iloc[0]["model"]
best_stacking_model = stacking_models[best_stacking_name]
best_stacking_predictions = stacking_predictions[best_stacking_name]
best_stacking_cm = stacking_confusion_matrices[best_stacking_name]

print("Best Stacking model:", best_stacking_name)
print(best_stacking_cm)

# Confusion matrix for best Stacking
plt.figure(figsize=(6, 5))
sns.heatmap(best_stacking_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Lower Risk", "Higher Risk"],
            yticklabels=["Lower Risk", "Higher Risk"])
plt.title("Stacking Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("../figures/german_stacking_confusion_matrix.png", dpi=200)
plt.show()


# ----- ENSEMBLE MODEL COMPARISON -----

ensemble_comparison = pd.DataFrame({
    "model_type": ["Random Forest", "AdaBoost", "Stacking"],
    "best_model": [best_random_forest_name, best_adaboost_name, best_stacking_name],
    "accuracy": [
        accuracy_score(y_ensemble_test, best_random_forest_predictions),
        accuracy_score(y_ensemble_test, best_adaboost_predictions),
        accuracy_score(y_ensemble_scaled_test, best_stacking_predictions)
    ],
    "precision": [
        precision_score(y_ensemble_test, best_random_forest_predictions, zero_division=0),
        precision_score(y_ensemble_test, best_adaboost_predictions, zero_division=0),
        precision_score(y_ensemble_scaled_test, best_stacking_predictions, zero_division=0)
    ],
    "recall": [
        recall_score(y_ensemble_test, best_random_forest_predictions, zero_division=0),
        recall_score(y_ensemble_test, best_adaboost_predictions, zero_division=0),
        recall_score(y_ensemble_scaled_test, best_stacking_predictions, zero_division=0)
    ],
    "f1_score": [
        f1_score(y_ensemble_test, best_random_forest_predictions, zero_division=0),
        f1_score(y_ensemble_test, best_adaboost_predictions, zero_division=0),
        f1_score(y_ensemble_scaled_test, best_stacking_predictions, zero_division=0)
    ],
    "true_lower_risk": [
        best_random_forest_cm[0, 0],
        best_adaboost_cm[0, 0],
        best_stacking_cm[0, 0]
    ],
    "lower_risk_predicted_higher_risk": [
        best_random_forest_cm[0, 1],
        best_adaboost_cm[0, 1],
        best_stacking_cm[0, 1]
    ],
    "higher_risk_predicted_lower_risk": [
        best_random_forest_cm[1, 0],
        best_adaboost_cm[1, 0],
        best_stacking_cm[1, 0]
    ],
    "true_higher_risk": [
        best_random_forest_cm[1, 1],
        best_adaboost_cm[1, 1],
        best_stacking_cm[1, 1]
    ]
})

ensemble_comparison = ensemble_comparison.round(4)
ensemble_comparison = ensemble_comparison.sort_values(["f1_score", "accuracy"], ascending=False)

print(ensemble_comparison)

ensemble_comparison.to_csv("../cleaned_data/german_ensemble_model_comparison.csv", index=False)

# Create image of ensemble model comparison
ensemble_comparison_preview = ensemble_comparison[[
    "model_type", "best_model", "accuracy", "precision", "recall", "f1_score",
    "true_lower_risk", "true_higher_risk"
]].copy()

plt.figure(figsize=(16, 3.2))
plt.axis("off")
table = plt.table(cellText=ensemble_comparison_preview.values,
                  colLabels=ensemble_comparison_preview.columns,
                  loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.4)
plt.title("German Credit Ensemble Model Comparison")
plt.tight_layout()
plt.savefig("../figures/german_ensemble_model_comparison.png", dpi=200, bbox_inches="tight")
plt.show()


# Ensemble metric comparison plot
ensemble_metric_plot = ensemble_comparison[["model_type", "accuracy", "precision", "recall", "f1_score"]].copy()
ensemble_metric_plot = ensemble_metric_plot.set_index("model_type")

ensemble_metric_plot.plot(kind="bar", figsize=(10, 6))
plt.title("German Credit Ensemble Model Metrics")
plt.xlabel("Model Type")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../figures/german_ensemble_model_metrics.png", dpi=200)
plt.show()


# ----- FINAL ENSEMBLE SUMMARY -----

final_ensemble_summary = pd.DataFrame({
    "item": [
        "records_used",
        "features_used",
        "training_records",
        "testing_records",
        "random_forest_settings_tested",
        "adaboost_settings_tested",
        "stacking_models_tested",
        "best_ensemble_model",
        "best_ensemble_accuracy",
        "best_ensemble_precision",
        "best_ensemble_recall",
        "best_ensemble_f1_score"
    ],
    "value": [
        prepared_ensemble.shape[0],
        x_ensemble.shape[1],
        x_ensemble_train.shape[0],
        x_ensemble_test.shape[0],
        len(random_forest_settings),
        len(adaboost_settings),
        len(stacking_settings),
        ensemble_comparison.iloc[0]["model_type"],
        ensemble_comparison.iloc[0]["accuracy"],
        ensemble_comparison.iloc[0]["precision"],
        ensemble_comparison.iloc[0]["recall"],
        ensemble_comparison.iloc[0]["f1_score"]
    ]
})

print(final_ensemble_summary)

final_ensemble_summary.to_csv("../cleaned_data/german_ensemble_final_summary.csv", index=False)

plt.figure(figsize=(12, 4))
plt.axis("off")
table = plt.table(cellText=final_ensemble_summary.values,
                  colLabels=final_ensemble_summary.columns,
                  loc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.4)
plt.title("Final Ensemble Classification Summary")
plt.tight_layout()
plt.savefig("../figures/german_ensemble_final_summary.png", dpi=200, bbox_inches="tight")
plt.show()
