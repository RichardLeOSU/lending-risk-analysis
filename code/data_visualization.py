import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
german = pd.read_csv("../cleaned_data/german_credit_cleaned.csv")
lending = pd.read_csv("../cleaned_data/lending_club_cleaned.csv")
cfpb = pd.read_csv("../cleaned_data/cfpb_cleaned.csv")

# ----- GERMAN CREDIT RISK ----

# Visualization 1: Age Histogram
plt.figure(figsize=(8, 5))
plt.hist(german["age"], bins=15, edgecolor="black")
plt.title("Distribution of Borrower Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("../figures/german_age_histogram.png")
plt.show()

# Visualization 2: Credit Amount Boxplot
plt.figure(figsize=(8, 5))
plt.boxplot(german["credit_amount"], vert=True)
plt.title("Boxplot of Credit Amount")
plt.ylabel("Credit Amount")
plt.tight_layout()
plt.savefig("../figures/german_credit_amount_boxplot.png")
plt.show()

# Visualization 3: Credit Amount Histogram
plt.figure(figsize=(8, 5))
plt.hist(german["credit_amount_per_month"].dropna(), bins=20, edgecolor="black")
plt.title("Distribution of Credit Amount Per Month")
plt.xlabel("Credit Amount Per Month")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("../figures/german_credit_amount_histogram.png")
plt.show()


# ----- LENDING CLUB ----

# Visualization 1: Loan Status Distribution
status_counts = lending["loan_status"].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(status_counts.index, status_counts.values)
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("../figures/lending_status_distribution.png")
plt.show()

# Visualization 2: Loan Amount by Loan Status
top_statuses = lending["loan_status"].value_counts().head(6).index
lending_top = lending[lending["loan_status"].isin(top_statuses)]

grouped_loan_amount = [
    lending_top[lending_top["loan_status"] == status]["loan_amount"]
    for status in top_statuses
]

plt.figure(figsize=(10, 6))
plt.boxplot(grouped_loan_amount, labels=top_statuses)
plt.title("Loan Amount by Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Loan Amount")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("../figures/lending_amount_by_status.png")
plt.show()

# Visualization 3: Annual Income by Loan Status
grouped_income = [
    lending_top[lending_top["loan_status"] == status]["annual_income"]
    for status in top_statuses
]

plt.figure(figsize=(10, 6))
plt.boxplot(grouped_income, labels=top_statuses)
plt.title("Annual Income by Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Annual Income")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("../figures/lending_income_or_dti_by_status.png")
plt.show()


# ----- CFPB ----
# Convert date_received back to datetime
cfpb["date_received"] = pd.to_datetime(cfpb["date_received"], format="%Y-%m-%d", errors="coerce")

# Visualization 1: Mortgage Complaint Volume by Month
cfpb["month"] = cfpb["date_received"].dt.to_period("M").astype(str)
monthly_counts = cfpb["month"].value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.plot(monthly_counts.index, monthly_counts.values, marker="o")
plt.title("CFPB Mortgage Complaint Volume by Month")
plt.xlabel("Month")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("../figures/cfpb_mortgage_complaints_by_month.png")
plt.show()

# Visualization 2: Mortgage Complaint Counts by Sub-Product
subproduct_counts = cfpb["sub_product"].value_counts().head(10).sort_values()

plt.figure(figsize=(10, 6))
plt.barh(subproduct_counts.index, subproduct_counts.values)
plt.title("CFPB Mortgage Complaint Counts by Sub-Product")
plt.xlabel("Count")
plt.ylabel("Mortgage Sub-Product")
plt.tight_layout()
plt.savefig("../figures/cfpb_mortgage_subproduct_counts.png")
plt.show()

# Visualization 3: Top Mortgage Complaint Issues
issue_counts = cfpb["issue"].value_counts().head(10).sort_values()

plt.figure(figsize=(10, 6))
plt.barh(issue_counts.index, issue_counts.values)
plt.title("Top Mortgage Complaint Issues in the CFPB Dataset")
plt.xlabel("Count")
plt.ylabel("Complaint Issue")
plt.tight_layout()
plt.savefig("../figures/cfpb_top_mortgage_issues.png")
plt.show()
