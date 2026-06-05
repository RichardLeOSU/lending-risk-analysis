import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB, BernoulliNB


# Load cleaned datasets
german = pd.read_csv("../cleaned_data/german_credit_cleaned.csv")
cfpb = pd.read_csv("../cleaned_data/cfpb_cleaned.csv")


# ----- DECISION TREE CLASSIFICATION: GERMAN CREDIT -----

# Keep useful columns for Decision Tree Classification
numeric_columns = ["age", "credit_amount", "duration", "credit_amount_per_month"]
categorical_columns = ["sex", "job", "housing", "saving_accounts", "checking_account", "purpose"]

columns_to_keep = numeric_columns + categorical_columns + ["credit_risk"]

german_tree = german[columns_to_keep].copy()

# Create binary label
# credit_risk = 1 stays 0 for lower risk
# credit_risk = 2 becomes 1 for higher risk
german_tree["bad_credit"] = np.where(german_tree["credit_risk"] == 2, 1, 0)

# Make sure numeric columns are numeric
for col in numeric_columns:
    german_tree[col] = pd.to_numeric(german_tree[col], errors="coerce")

# Replace missing numeric values with median
german_tree[numeric_columns] = german_tree[numeric_columns].replace([np.inf, -np.inf], np.nan)
german_tree[numeric_columns] = german_tree[numeric_columns].fillna(german_tree[numeric_columns].median())

# Replace missing categorical values with unknown
for col in categorical_columns:
    german_tree[col] = german_tree[col].fillna("unknown").astype(str)

# Convert categorical variables into dummy columns
x = german_tree[numeric_columns + categorical_columns].copy()
x = pd.get_dummies(x, columns=categorical_columns, drop_first=True)

# Convert boolean dummy columns to integers
dummy_columns = x.select_dtypes(include="bool").columns
x[dummy_columns] = x[dummy_columns].astype(int)

y = german_tree["bad_credit"]

# Final checks
prepared_tree = x.copy()
prepared_tree["bad_credit"] = y.values

print(prepared_tree.head())
print(prepared_tree.info())
print(prepared_tree.isnull().sum())
print("Duplicate rows:", prepared_tree.duplicated().sum())

# Save prepared Decision Tree dataset
prepared_tree.to_csv("../cleaned_data/german_decision_tree_prepared.csv", index=False)


# Class distribution
class_counts = prepared_tree["bad_credit"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(["Lower Risk", "Higher Risk"], class_counts.values)
plt.title("German Credit Risk Distribution for Decision Tree Classification")
plt.xlabel("Credit Risk Group")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../figures/german_decision_tree_class_distribution.png", dpi=200)
plt.show()


# Split data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=42, stratify=y
)

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# ----- DECISION TREE MODEL 1: GINI, ALL FEATURES -----

tree_model_1 = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

tree_model_1.fit(x_train, y_train)

# Make predictions
y_test_pred_1 = tree_model_1.predict(x_test)


# ----- DECISION TREE MODEL 2: ENTROPY, ALL FEATURES -----

tree_model_2 = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    min_samples_leaf=15,
    class_weight="balanced",
    random_state=42
)

tree_model_2.fit(x_train, y_train)

# Make predictions
y_test_pred_2 = tree_model_2.predict(x_test)


# ----- DECISION TREE MODEL 3: GINI, SELECTED FEATURES -----

# Use a smaller set of important lending-risk variables
selected_features = [
    "age", "credit_amount", "duration", "credit_amount_per_month",
    "checking_account_moderate", "checking_account_rich", "checking_account_unknown",
    "saving_accounts_unknown", "housing_own", "purpose_radio/TV"
]

# Keep only selected features that exist in the prepared dataset
selected_features = [col for col in selected_features if col in x.columns]

x_selected = x[selected_features].copy()

x_train_selected, x_test_selected, y_train_selected, y_test_selected = train_test_split(
    x_selected, y, test_size=0.30, random_state=42, stratify=y
)

tree_model_3 = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

tree_model_3.fit(x_train_selected, y_train_selected)

# Make predictions
y_test_pred_3 = tree_model_3.predict(x_test_selected)


# ----- DECISION TREE MODEL COMPARISON -----

tree_results = pd.DataFrame({
    "model": ["tree_1_gini_all_features", "tree_2_entropy_all_features", "tree_3_gini_selected_features"],
    "criterion": ["gini", "entropy", "gini"],
    "max_depth": [3, 4, 4],
    "min_samples_leaf": [10, 15, 10],
    "features_used": [x.shape[1], x.shape[1], len(selected_features)],
    "accuracy": [
        accuracy_score(y_test, y_test_pred_1),
        accuracy_score(y_test, y_test_pred_2),
        accuracy_score(y_test_selected, y_test_pred_3)
    ],
    "precision": [
        precision_score(y_test, y_test_pred_1),
        precision_score(y_test, y_test_pred_2),
        precision_score(y_test_selected, y_test_pred_3)
    ],
    "recall": [
        recall_score(y_test, y_test_pred_1),
        recall_score(y_test, y_test_pred_2),
        recall_score(y_test_selected, y_test_pred_3)
    ],
    "f1_score": [
        f1_score(y_test, y_test_pred_1),
        f1_score(y_test, y_test_pred_2),
        f1_score(y_test_selected, y_test_pred_3)
    ]
})

tree_results = tree_results.round(4)

print(tree_results)

# Save model comparison results
tree_results.to_csv("../cleaned_data/german_decision_tree_model_results.csv", index=False)

# Create image of model comparison results
plt.figure(figsize=(13, 2.8))
plt.axis("off")
plt.table(cellText=tree_results.values,
          colLabels=tree_results.columns,
          loc="center")
plt.title("German Credit Decision Tree Model Comparison")
plt.tight_layout()
plt.savefig("../figures/german_decision_tree_model_comparison.png", dpi=200)
plt.show()


# ----- DECISION TREE VISUALIZATIONS -----

# Create tree image for Model 1
plt.figure(figsize=(18, 9))
plot_tree(tree_model_1,
          feature_names=x.columns,
          class_names=["Lower Risk", "Higher Risk"],
          rounded=True,
          fontsize=8)
plt.title("Decision Tree Model 1: Gini, All Features")
plt.tight_layout()
plt.savefig("../figures/german_decision_tree_model_1.png", dpi=200)
plt.show()


# Create tree image for Model 2
plt.figure(figsize=(22, 11))
plot_tree(tree_model_2,
          feature_names=x.columns,
          class_names=["Lower Risk", "Higher Risk"],
          rounded=True,
          fontsize=7)
plt.title("Decision Tree Model 2: Entropy, All Features")
plt.tight_layout()
plt.savefig("../figures/german_decision_tree_model_2.png", dpi=200)
plt.show()


# Create tree image for Model 3
plt.figure(figsize=(22, 11))
plot_tree(tree_model_3,
          feature_names=selected_features,
          class_names=["Lower Risk", "Higher Risk"],
          rounded=True,
          fontsize=7)
plt.title("Decision Tree Model 3: Gini, Selected Features")
plt.tight_layout()
plt.savefig("../figures/german_decision_tree_model_3.png", dpi=200)
plt.show()


# ----- DECISION TREE FEATURE IMPORTANCE -----

# Use Model 2 for feature importance because it used all features and a slightly deeper tree
feature_importance = pd.DataFrame({
    "feature": x.columns,
    "importance": tree_model_2.feature_importances_
})

feature_importance = feature_importance.sort_values("importance", ascending=False)
feature_importance = feature_importance[feature_importance["importance"] > 0]
feature_importance["importance"] = feature_importance["importance"].round(4)

print(feature_importance)

# Save feature importance results
feature_importance.to_csv("../cleaned_data/german_decision_tree_feature_importance.csv", index=False)

# Create image of feature importance table
feature_importance_preview = feature_importance.head(12).copy()

plt.figure(figsize=(10, 4))
plt.axis("off")
plt.table(cellText=feature_importance_preview.values,
          colLabels=feature_importance_preview.columns,
          loc="center")
plt.title("Top Decision Tree Feature Importances")
plt.tight_layout()
plt.savefig("../figures/german_decision_tree_feature_importance_table.png", dpi=200)
plt.show()


# Create feature importance plot
top_features = feature_importance.head(12).sort_values("importance")

plt.figure(figsize=(9, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.title("Top Decision Tree Feature Importances")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("../figures/german_decision_tree_feature_importance.png", dpi=200)
plt.show()


# ----- DECISION TREE CONFUSION MATRICES -----

# Confusion matrix for Tree 1
cm_tree_1 = confusion_matrix(y_test, y_test_pred_1)

print("Decision Tree Model 1 Confusion Matrix")
print(cm_tree_1)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_tree_1, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Lower Risk", "Higher Risk"],
            yticklabels=["Lower Risk", "Higher Risk"])
plt.title("Decision Tree Model 1 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("../figures/german_decision_tree_model_1_confusion_matrix.png", dpi=200)
plt.show()


# Confusion matrix for Tree 2
cm_tree_2 = confusion_matrix(y_test, y_test_pred_2)

print("Decision Tree Model 2 Confusion Matrix")
print(cm_tree_2)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_tree_2, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Lower Risk", "Higher Risk"],
            yticklabels=["Lower Risk", "Higher Risk"])
plt.title("Decision Tree Model 2 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("../figures/german_decision_tree_model_2_confusion_matrix.png", dpi=200)
plt.show()


# Confusion matrix for Tree 3
cm_tree_3 = confusion_matrix(y_test_selected, y_test_pred_3)

print("Decision Tree Model 3 Confusion Matrix")
print(cm_tree_3)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_tree_3, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Lower Risk", "Higher Risk"],
            yticklabels=["Lower Risk", "Higher Risk"])
plt.title("Decision Tree Model 3 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("../figures/german_decision_tree_model_3_confusion_matrix.png", dpi=200)
plt.show()


# ----- DECISION TREE CONFUSION MATRIX SUMMARY -----

tree_confusion_summary = pd.DataFrame({
    "model": ["tree_1_gini_all_features", "tree_2_entropy_all_features", "tree_3_gini_selected_features"],
    "true_lower_risk": [
        cm_tree_1[0, 0],
        cm_tree_2[0, 0],
        cm_tree_3[0, 0]
    ],
    "lower_risk_predicted_higher_risk": [
        cm_tree_1[0, 1],
        cm_tree_2[0, 1],
        cm_tree_3[0, 1]
    ],
    "higher_risk_predicted_lower_risk": [
        cm_tree_1[1, 0],
        cm_tree_2[1, 0],
        cm_tree_3[1, 0]
    ],
    "true_higher_risk": [
        cm_tree_1[1, 1],
        cm_tree_2[1, 1],
        cm_tree_3[1, 1]
    ],
    "accuracy": [
        accuracy_score(y_test, y_test_pred_1),
        accuracy_score(y_test, y_test_pred_2),
        accuracy_score(y_test_selected, y_test_pred_3)
    ],
    "precision": [
        precision_score(y_test, y_test_pred_1),
        precision_score(y_test, y_test_pred_2),
        precision_score(y_test_selected, y_test_pred_3)
    ],
    "recall": [
        recall_score(y_test, y_test_pred_1),
        recall_score(y_test, y_test_pred_2),
        recall_score(y_test_selected, y_test_pred_3)
    ],
    "f1_score": [
        f1_score(y_test, y_test_pred_1),
        f1_score(y_test, y_test_pred_2),
        f1_score(y_test_selected, y_test_pred_3)
    ]
})

tree_confusion_summary = tree_confusion_summary.round(4)

print(tree_confusion_summary)

# Save confusion matrix summary
tree_confusion_summary.to_csv("../cleaned_data/german_decision_tree_confusion_summary.csv", index=False)

# Create image of confusion matrix summary
plt.figure(figsize=(14, 3))
plt.axis("off")
plt.table(cellText=tree_confusion_summary.values,
          colLabels=tree_confusion_summary.columns,
          loc="center")
plt.title("German Credit Decision Tree Confusion Matrix Summary")
plt.tight_layout()
plt.savefig("../figures/german_decision_tree_confusion_summary.png", dpi=200)
plt.show()


# Save predictions for each tree
tree_predictions = pd.DataFrame({
    "actual_bad_credit": y_test,
    "tree_1_predicted_bad_credit": y_test_pred_1,
    "tree_2_predicted_bad_credit": y_test_pred_2
})

tree_predictions_selected = pd.DataFrame({
    "actual_bad_credit": y_test_selected,
    "tree_3_predicted_bad_credit": y_test_pred_3
})

tree_predictions.to_csv("../cleaned_data/german_decision_tree_predictions_models_1_and_2.csv", index=False)
tree_predictions_selected.to_csv("../cleaned_data/german_decision_tree_predictions_model_3.csv", index=False)


# ----- NAIVE BAYES CLASSIFICATION -----


# ----- GERMAN CREDIT TEXT PREPARATION FOR GAUSSIAN AND CATEGORICAL NAIVE BAYES -----

german_nb = german.copy()

# Create binary label
# credit_risk = 1 stays 0 for lower risk
# credit_risk = 2 becomes 1 for higher risk
german_nb["bad_credit"] = np.where(german_nb["credit_risk"] == 2, 1, 0)


# ----- GAUSSIAN NAIVE BAYES: GERMAN CREDIT NUMERIC DATA -----

# Keep continuous numeric columns for Gaussian Naive Bayes
gaussian_columns = ["age", "credit_amount", "duration", "credit_amount_per_month"]

german_gaussian_nb = german_nb[gaussian_columns + ["bad_credit"]].copy()

# Make sure numeric columns are numeric
for col in gaussian_columns:
    german_gaussian_nb[col] = pd.to_numeric(german_gaussian_nb[col], errors="coerce")

# Replace missing values with median
german_gaussian_nb = german_gaussian_nb.replace([np.inf, -np.inf], np.nan)
german_gaussian_nb[gaussian_columns] = german_gaussian_nb[gaussian_columns].fillna(
    german_gaussian_nb[gaussian_columns].median()
)

# Final checks
print(german_gaussian_nb.head())
print(german_gaussian_nb.info())
print(german_gaussian_nb.isnull().sum())
print("Duplicate rows:", german_gaussian_nb.duplicated().sum())

# Save prepared Gaussian Naive Bayes dataset
german_gaussian_nb.to_csv("../cleaned_data/german_gaussian_nb_prepared.csv", index=False)


# ----- CATEGORICAL NAIVE BAYES: GERMAN CREDIT CATEGORICAL DATA -----

# Keep categorical columns for Categorical Naive Bayes
categorical_columns = ["sex", "job", "housing", "saving_accounts", "checking_account", "purpose"]

german_categorical_nb = german_nb[categorical_columns + ["bad_credit"]].copy()

# Replace missing categorical values and encode categories as numbers
for col in categorical_columns:
    german_categorical_nb[col] = german_categorical_nb[col].fillna("unknown").astype(str)
    german_categorical_nb[col] = german_categorical_nb[col].astype("category").cat.codes

# Final checks
print(german_categorical_nb.head())
print(german_categorical_nb.info())
print(german_categorical_nb.isnull().sum())
print("Duplicate rows:", german_categorical_nb.duplicated().sum())

# Save prepared Categorical Naive Bayes dataset
german_categorical_nb.to_csv("../cleaned_data/german_categorical_nb_prepared.csv", index=False)


# ----- CFPB TEXT PREPARATION FOR MULTINOMIAL AND BERNOULLI NAIVE BAYES -----

# Keep the top four mortgage sub-products to avoid very small classes
top_sub_products = cfpb["sub_product"].value_counts().head(4).index.tolist()

cfpb_nb = cfpb[cfpb["sub_product"].isin(top_sub_products)].copy()

# Combine useful complaint fields into one text feature
cfpb_nb["text_features"] = (
    cfpb_nb["issue"].fillna("").astype(str) + " " +
    cfpb_nb["sub_issue"].fillna("").astype(str) + " " +
    cfpb_nb["company_response_to_consumer"].fillna("").astype(str) + " " +
    cfpb_nb["submitted_via"].fillna("").astype(str)
)

# Create label
cfpb_nb["sub_product_label"] = cfpb_nb["sub_product"].astype(str)

# Use a limited vocabulary so the prepared dataset is understandable in the report
text_vectorizer = CountVectorizer(stop_words="english", max_features=20)

cfpb_word_counts = text_vectorizer.fit_transform(cfpb_nb["text_features"])
word_features = text_vectorizer.get_feature_names_out()


# ----- MULTINOMIAL NAIVE BAYES: CFPB WORD COUNT DATA -----

cfpb_multinomial_nb = pd.DataFrame(cfpb_word_counts.toarray(), columns=word_features)
cfpb_multinomial_nb["sub_product_label"] = cfpb_nb["sub_product_label"].values

# Final checks
print(cfpb_multinomial_nb.head())
print(cfpb_multinomial_nb.info())
print(cfpb_multinomial_nb.isnull().sum())
print("Duplicate rows:", cfpb_multinomial_nb.duplicated().sum())

# Save prepared Multinomial Naive Bayes dataset
cfpb_multinomial_nb.to_csv("../cleaned_data/cfpb_multinomial_nb_prepared.csv", index=False)


# ----- BERNOULLI NAIVE BAYES: CFPB BINARY WORD PRESENCE DATA -----

cfpb_binary_values = (cfpb_word_counts.toarray() > 0).astype(int)

cfpb_bernoulli_nb = pd.DataFrame(cfpb_binary_values, columns=word_features)
cfpb_bernoulli_nb["sub_product_label"] = cfpb_nb["sub_product_label"].values

# Final checks
print(cfpb_bernoulli_nb.head())
print(cfpb_bernoulli_nb.info())
print(cfpb_bernoulli_nb.isnull().sum())
print("Duplicate rows:", cfpb_bernoulli_nb.duplicated().sum())

# Save prepared Bernoulli Naive Bayes dataset
cfpb_bernoulli_nb.to_csv("../cleaned_data/cfpb_bernoulli_nb_prepared.csv", index=False)


# ----- NAIVE BAYES DATA PREPARATION PLAN -----

naive_bayes_plan = pd.DataFrame({
    "method": ["GaussianNB", "CategoricalNB", "MultinomialNB", "BernoulliNB"],
    "dataset": ["German Credit", "German Credit", "CFPB Complaints", "CFPB Complaints"],
    "format": ["continuous numeric", "encoded categories", "word counts", "binary word presence"],
    "label": ["bad_credit", "bad_credit", "sub_product_label", "sub_product_label"],
    "rows": [
        german_gaussian_nb.shape[0],
        german_categorical_nb.shape[0],
        cfpb_multinomial_nb.shape[0],
        cfpb_bernoulli_nb.shape[0]
    ],
    "features": [
        german_gaussian_nb.shape[1] - 1,
        german_categorical_nb.shape[1] - 1,
        cfpb_multinomial_nb.shape[1] - 1,
        cfpb_bernoulli_nb.shape[1] - 1
    ]
})

print(naive_bayes_plan)

# Save Naive Bayes preparation plan
naive_bayes_plan.to_csv("../cleaned_data/naive_bayes_data_preparation_plan.csv", index=False)

# Create image of Naive Bayes preparation plan
plt.figure(figsize=(12, 3))
plt.axis("off")
plt.table(cellText=naive_bayes_plan.values,
          colLabels=naive_bayes_plan.columns,
          loc="center")
plt.title("Naive Bayes Data Preparation Plan")
plt.tight_layout()
plt.savefig("../figures/naive_bayes_data_preparation_plan.png", dpi=200)
plt.show()


# CFPB class distribution used for text-based Naive Bayes models
cfpb_sub_product_counts = cfpb_nb["sub_product_label"].value_counts()

print(cfpb_sub_product_counts)

plt.figure(figsize=(9, 5))
plt.barh(cfpb_sub_product_counts.index, cfpb_sub_product_counts.values)
plt.title("CFPB Sub-Product Distribution for Naive Bayes")
plt.xlabel("Count")
plt.ylabel("Sub-Product")
plt.tight_layout()
plt.savefig("../figures/cfpb_naive_bayes_sub_product_distribution.png", dpi=200)
plt.show()


# ----- GAUSSIAN NAIVE BAYES: GERMAN CREDIT MODELING AND RESULTS -----

# Set independent variables and dependent variable
x_gaussian = german_gaussian_nb.drop(columns=["bad_credit"])
y_gaussian = german_gaussian_nb["bad_credit"]

# Split data into training and testing data
x_gaussian_train, x_gaussian_test, y_gaussian_train, y_gaussian_test = train_test_split(
    x_gaussian, y_gaussian, test_size=0.30, random_state=42, stratify=y_gaussian
)

print("Gaussian NB training data shape:", x_gaussian_train.shape)
print("Gaussian NB testing data shape:", x_gaussian_test.shape)

# Run Gaussian Naive Bayes
gaussian_nb_model = GaussianNB()
gaussian_nb_model.fit(x_gaussian_train, y_gaussian_train)

# Make predictions
gaussian_predictions = gaussian_nb_model.predict(x_gaussian_test)
gaussian_probabilities = gaussian_nb_model.predict_proba(x_gaussian_test)

print("Gaussian NB classes:", gaussian_nb_model.classes_)
print("Gaussian NB probabilities:")
print(np.round(gaussian_probabilities[:10], 4))


# ----- CATEGORICAL NAIVE BAYES: GERMAN CREDIT MODELING AND RESULTS -----

# Set independent variables and dependent variable
x_categorical = german_categorical_nb.drop(columns=["bad_credit"])
y_categorical = german_categorical_nb["bad_credit"]

# Split data into training and testing data
x_categorical_train, x_categorical_test, y_categorical_train, y_categorical_test = train_test_split(
    x_categorical, y_categorical, test_size=0.30, random_state=42, stratify=y_categorical
)

print("Categorical NB training data shape:", x_categorical_train.shape)
print("Categorical NB testing data shape:", x_categorical_test.shape)

# Run Categorical Naive Bayes
categorical_nb_model = CategoricalNB()
categorical_nb_model.fit(x_categorical_train, y_categorical_train)

# Make predictions
categorical_predictions = categorical_nb_model.predict(x_categorical_test)
categorical_probabilities = categorical_nb_model.predict_proba(x_categorical_test)

print("Categorical NB classes:", categorical_nb_model.classes_)
print("Categorical NB probabilities:")
print(np.round(categorical_probabilities[:10], 4))


# ----- MULTINOMIAL NAIVE BAYES: CFPB COMPLAINTS MODELING AND RESULTS -----

# Set independent variables and dependent variable
x_multinomial = cfpb_multinomial_nb.drop(columns=["sub_product_label"])
y_multinomial = cfpb_multinomial_nb["sub_product_label"]

# Split data into training and testing data
x_multinomial_train, x_multinomial_test, y_multinomial_train, y_multinomial_test = train_test_split(
    x_multinomial, y_multinomial, test_size=0.30, random_state=42, stratify=y_multinomial
)

print("Multinomial NB training data shape:", x_multinomial_train.shape)
print("Multinomial NB testing data shape:", x_multinomial_test.shape)

# Run Multinomial Naive Bayes
multinomial_nb_model = MultinomialNB()
multinomial_nb_model.fit(x_multinomial_train, y_multinomial_train)

# Make predictions
multinomial_predictions = multinomial_nb_model.predict(x_multinomial_test)
multinomial_probabilities = multinomial_nb_model.predict_proba(x_multinomial_test)

print("Multinomial NB classes:", multinomial_nb_model.classes_)
print("Multinomial NB probabilities:")
print(np.round(multinomial_probabilities[:10], 4))


# ----- BERNOULLI NAIVE BAYES: CFPB COMPLAINTS MODELING AND RESULTS -----

# Set independent variables and dependent variable
x_bernoulli = cfpb_bernoulli_nb.drop(columns=["sub_product_label"])
y_bernoulli = cfpb_bernoulli_nb["sub_product_label"]

# Split data into training and testing data
x_bernoulli_train, x_bernoulli_test, y_bernoulli_train, y_bernoulli_test = train_test_split(
    x_bernoulli, y_bernoulli, test_size=0.30, random_state=42, stratify=y_bernoulli
)

print("Bernoulli NB training data shape:", x_bernoulli_train.shape)
print("Bernoulli NB testing data shape:", x_bernoulli_test.shape)

# Run Bernoulli Naive Bayes
bernoulli_nb_model = BernoulliNB()
bernoulli_nb_model.fit(x_bernoulli_train, y_bernoulli_train)

# Make predictions
bernoulli_predictions = bernoulli_nb_model.predict(x_bernoulli_test)
bernoulli_probabilities = bernoulli_nb_model.predict_proba(x_bernoulli_test)

print("Bernoulli NB classes:", bernoulli_nb_model.classes_)
print("Bernoulli NB probabilities:")
print(np.round(bernoulli_probabilities[:10], 4))


# ----- NAIVE BAYES MODEL COMPARISON -----

naive_bayes_results = pd.DataFrame({
    "model": ["GaussianNB", "CategoricalNB", "MultinomialNB", "BernoulliNB"],
    "dataset": ["German Credit", "German Credit", "CFPB Complaints", "CFPB Complaints"],
    "target_label": ["bad_credit", "bad_credit", "sub_product_label", "sub_product_label"],
    "accuracy": [
        accuracy_score(y_gaussian_test, gaussian_predictions),
        accuracy_score(y_categorical_test, categorical_predictions),
        accuracy_score(y_multinomial_test, multinomial_predictions),
        accuracy_score(y_bernoulli_test, bernoulli_predictions)
    ],
    "precision_weighted": [
        precision_score(y_gaussian_test, gaussian_predictions, average="weighted", zero_division=0),
        precision_score(y_categorical_test, categorical_predictions, average="weighted", zero_division=0),
        precision_score(y_multinomial_test, multinomial_predictions, average="weighted", zero_division=0),
        precision_score(y_bernoulli_test, bernoulli_predictions, average="weighted", zero_division=0)
    ],
    "recall_weighted": [
        recall_score(y_gaussian_test, gaussian_predictions, average="weighted", zero_division=0),
        recall_score(y_categorical_test, categorical_predictions, average="weighted", zero_division=0),
        recall_score(y_multinomial_test, multinomial_predictions, average="weighted", zero_division=0),
        recall_score(y_bernoulli_test, bernoulli_predictions, average="weighted", zero_division=0)
    ],
    "f1_score_weighted": [
        f1_score(y_gaussian_test, gaussian_predictions, average="weighted", zero_division=0),
        f1_score(y_categorical_test, categorical_predictions, average="weighted", zero_division=0),
        f1_score(y_multinomial_test, multinomial_predictions, average="weighted", zero_division=0),
        f1_score(y_bernoulli_test, bernoulli_predictions, average="weighted", zero_division=0)
    ]
})

naive_bayes_results = naive_bayes_results.round(4)

print(naive_bayes_results)

# Save Naive Bayes model results
naive_bayes_results.to_csv("../cleaned_data/naive_bayes_model_results.csv", index=False)

# Create image of Naive Bayes model results
plt.figure(figsize=(13, 3))
plt.axis("off")
plt.table(cellText=naive_bayes_results.values,
          colLabels=naive_bayes_results.columns,
          loc="center")
plt.title("Naive Bayes Model Results")
plt.tight_layout()
plt.savefig("../figures/naive_bayes_model_results.png", dpi=200)
plt.show()


# ----- NAIVE BAYES PREDICTION PROBABILITIES -----

# Probability table for Gaussian Naive Bayes
gaussian_probability_table = pd.DataFrame(
    gaussian_probabilities,
    columns=["prob_lower_risk", "prob_higher_risk"],
    index=x_gaussian_test.index
)

gaussian_probability_table["actual_bad_credit"] = y_gaussian_test.values
gaussian_probability_table["predicted_bad_credit"] = gaussian_predictions

gaussian_probability_table = gaussian_probability_table[[
    "actual_bad_credit", "predicted_bad_credit", "prob_lower_risk", "prob_higher_risk"
]]

gaussian_probability_table = gaussian_probability_table.round(4)

print(gaussian_probability_table.head(10))

gaussian_probability_table.to_csv("../cleaned_data/gaussian_nb_prediction_probabilities.csv", index=False)


# Probability table for Categorical Naive Bayes
categorical_probability_table = pd.DataFrame(
    categorical_probabilities,
    columns=["prob_lower_risk", "prob_higher_risk"],
    index=x_categorical_test.index
)

categorical_probability_table["actual_bad_credit"] = y_categorical_test.values
categorical_probability_table["predicted_bad_credit"] = categorical_predictions

categorical_probability_table = categorical_probability_table[[
    "actual_bad_credit", "predicted_bad_credit", "prob_lower_risk", "prob_higher_risk"
]]

categorical_probability_table = categorical_probability_table.round(4)

print(categorical_probability_table.head(10))

categorical_probability_table.to_csv("../cleaned_data/categorical_nb_prediction_probabilities.csv", index=False)


# Probability table for Multinomial Naive Bayes
multinomial_probability_table = pd.DataFrame(
    multinomial_probabilities,
    columns=["prob_" + str(class_name) for class_name in multinomial_nb_model.classes_],
    index=x_multinomial_test.index
)

multinomial_probability_table["actual_sub_product"] = y_multinomial_test.values
multinomial_probability_table["predicted_sub_product"] = multinomial_predictions

multinomial_probability_table = multinomial_probability_table[
    ["actual_sub_product", "predicted_sub_product"] +
    ["prob_" + str(class_name) for class_name in multinomial_nb_model.classes_]
]

multinomial_probability_table = multinomial_probability_table.round(4)

print(multinomial_probability_table.head(10))

multinomial_probability_table.to_csv("../cleaned_data/multinomial_nb_prediction_probabilities.csv", index=False)


# Probability table for Bernoulli Naive Bayes
bernoulli_probability_table = pd.DataFrame(
    bernoulli_probabilities,
    columns=["prob_" + str(class_name) for class_name in bernoulli_nb_model.classes_],
    index=x_bernoulli_test.index
)

bernoulli_probability_table["actual_sub_product"] = y_bernoulli_test.values
bernoulli_probability_table["predicted_sub_product"] = bernoulli_predictions

bernoulli_probability_table = bernoulli_probability_table[
    ["actual_sub_product", "predicted_sub_product"] +
    ["prob_" + str(class_name) for class_name in bernoulli_nb_model.classes_]
]

bernoulli_probability_table = bernoulli_probability_table.round(4)

print(bernoulli_probability_table.head(10))

bernoulli_probability_table.to_csv("../cleaned_data/bernoulli_nb_prediction_probabilities.csv", index=False)


# Create image of Gaussian probability examples
gaussian_probability_preview = gaussian_probability_table.head(8).copy()

plt.figure(figsize=(10, 3))
plt.axis("off")
plt.table(cellText=gaussian_probability_preview.values,
          colLabels=gaussian_probability_preview.columns,
          loc="center")
plt.title("Gaussian Naive Bayes Prediction Probabilities")
plt.tight_layout()
plt.savefig("../figures/gaussian_nb_prediction_probabilities.png", dpi=200)
plt.show()


# Create image of Categorical probability examples
categorical_probability_preview = categorical_probability_table.head(8).copy()

plt.figure(figsize=(10, 3))
plt.axis("off")
plt.table(cellText=categorical_probability_preview.values,
          colLabels=categorical_probability_preview.columns,
          loc="center")
plt.title("Categorical Naive Bayes Prediction Probabilities")
plt.tight_layout()
plt.savefig("../figures/categorical_nb_prediction_probabilities.png", dpi=200)
plt.show()


# Create image of Multinomial probability examples
multinomial_probability_preview = multinomial_probability_table.head(6).copy()

plt.figure(figsize=(18, 3.5))
plt.axis("off")
plt.table(cellText=multinomial_probability_preview.values,
          colLabels=multinomial_probability_preview.columns,
          loc="center")
plt.title("Multinomial Naive Bayes Prediction Probabilities")
plt.tight_layout()
plt.savefig("../figures/multinomial_nb_prediction_probabilities.png", dpi=200)
plt.show()


# Create image of Bernoulli probability examples
bernoulli_probability_preview = bernoulli_probability_table.head(6).copy()

plt.figure(figsize=(18, 3.5))
plt.axis("off")
plt.table(cellText=bernoulli_probability_preview.values,
          colLabels=bernoulli_probability_preview.columns,
          loc="center")
plt.title("Bernoulli Naive Bayes Prediction Probabilities")
plt.tight_layout()
plt.savefig("../figures/bernoulli_nb_prediction_probabilities.png", dpi=200)
plt.show()


# ----- NAIVE BAYES CONFUSION MATRICES -----

# Gaussian Naive Bayes confusion matrix
cm_gaussian = confusion_matrix(y_gaussian_test, gaussian_predictions)

print("Gaussian Naive Bayes Confusion Matrix")
print(cm_gaussian)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_gaussian, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Lower Risk", "Higher Risk"],
            yticklabels=["Lower Risk", "Higher Risk"])
plt.title("Gaussian Naive Bayes Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("../figures/gaussian_nb_confusion_matrix.png", dpi=200)
plt.show()


# Categorical Naive Bayes confusion matrix
cm_categorical = confusion_matrix(y_categorical_test, categorical_predictions)

print("Categorical Naive Bayes Confusion Matrix")
print(cm_categorical)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_categorical, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Lower Risk", "Higher Risk"],
            yticklabels=["Lower Risk", "Higher Risk"])
plt.title("Categorical Naive Bayes Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("../figures/categorical_nb_confusion_matrix.png", dpi=200)
plt.show()


# Multinomial Naive Bayes confusion matrix
cm_multinomial = confusion_matrix(y_multinomial_test, multinomial_predictions,
                                  labels=multinomial_nb_model.classes_)

print("Multinomial Naive Bayes Confusion Matrix")
print(cm_multinomial)

plt.figure(figsize=(9, 7))
sns.heatmap(cm_multinomial, annot=True, fmt="d", cmap="Blues",
            xticklabels=multinomial_nb_model.classes_,
            yticklabels=multinomial_nb_model.classes_)
plt.title("Multinomial Naive Bayes Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("../figures/multinomial_nb_confusion_matrix.png", dpi=200)
plt.show()


# Bernoulli Naive Bayes confusion matrix
cm_bernoulli = confusion_matrix(y_bernoulli_test, bernoulli_predictions,
                                labels=bernoulli_nb_model.classes_)

print("Bernoulli Naive Bayes Confusion Matrix")
print(cm_bernoulli)

plt.figure(figsize=(9, 7))
sns.heatmap(cm_bernoulli, annot=True, fmt="d", cmap="Blues",
            xticklabels=bernoulli_nb_model.classes_,
            yticklabels=bernoulli_nb_model.classes_)
plt.title("Bernoulli Naive Bayes Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("../figures/bernoulli_nb_confusion_matrix.png", dpi=200)
plt.show()


# ----- NAIVE BAYES CONFUSION MATRIX SUMMARY -----

naive_bayes_confusion_summary = pd.DataFrame({
    "model": ["GaussianNB", "CategoricalNB", "MultinomialNB", "BernoulliNB"],
    "dataset": ["German Credit", "German Credit", "CFPB Complaints", "CFPB Complaints"],
    "correct_predictions": [
        np.trace(cm_gaussian),
        np.trace(cm_categorical),
        np.trace(cm_multinomial),
        np.trace(cm_bernoulli)
    ],
    "total_testing_records": [
        cm_gaussian.sum(),
        cm_categorical.sum(),
        cm_multinomial.sum(),
        cm_bernoulli.sum()
    ],
    "accuracy": [
        accuracy_score(y_gaussian_test, gaussian_predictions),
        accuracy_score(y_categorical_test, categorical_predictions),
        accuracy_score(y_multinomial_test, multinomial_predictions),
        accuracy_score(y_bernoulli_test, bernoulli_predictions)
    ],
    "weighted_f1_score": [
        f1_score(y_gaussian_test, gaussian_predictions, average="weighted", zero_division=0),
        f1_score(y_categorical_test, categorical_predictions, average="weighted", zero_division=0),
        f1_score(y_multinomial_test, multinomial_predictions, average="weighted", zero_division=0),
        f1_score(y_bernoulli_test, bernoulli_predictions, average="weighted", zero_division=0)
    ]
})

naive_bayes_confusion_summary = naive_bayes_confusion_summary.round(4)

print(naive_bayes_confusion_summary)

# Save confusion matrix summary
naive_bayes_confusion_summary.to_csv("../cleaned_data/naive_bayes_confusion_summary.csv", index=False)

# Create image of confusion matrix summary
plt.figure(figsize=(12, 3))
plt.axis("off")
plt.table(cellText=naive_bayes_confusion_summary.values,
          colLabels=naive_bayes_confusion_summary.columns,
          loc="center")
plt.title("Naive Bayes Confusion Matrix Summary")
plt.tight_layout()
plt.savefig("../figures/naive_bayes_confusion_summary.png", dpi=200)
plt.show()
