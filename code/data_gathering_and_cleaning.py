import pandas as pd
import numpy as np
import requests
from io import StringIO

german = pd.read_csv("../raw_data/german_credit_raw.csv")
lending = pd.read_csv("../raw_data/lending_club_raw.csv")


# ----- GERMAN CREDIT RISK ----

# Initial checks
# print(german.head())
# print(german.info())
# print(german.isnull().sum())
# print(german.duplicated().sum())

# Drop extra index column
german = german.drop(columns=["Unnamed: 0"])

# Clean column names
german.columns = (german.columns.str.strip().str.lower().str.replace(" ", "_"))

# Clean text columns
text_cols = ["sex", "housing", "saving_accounts", "checking_account", "purpose"]
for col in text_cols:
    german[col] = german[col].astype("string").str.strip().str.lower()

# Fill missing categorical values
german["saving_accounts"] = german["saving_accounts"].fillna("unknown")
german["checking_account"] = german["checking_account"].fillna("unknown")

# Treat job as categorical
german["job"] = german["job"].astype("category")

# Create a new quantitative feature
german["credit_amount_per_month"] = np.where(
    german["duration"] > 0,
    german["credit_amount"] / german["duration"],
    np.nan
)

# Final checks
print(german.head())
print(german.info())
print(german.isnull().sum())
print("Duplicate rows:", german.duplicated().sum())

# Save cleaned file
german.to_csv("../cleaned_data/german_credit_cleaned.csv", index=False)


# ----- LENDING CLUB ----

# Initial checks
# print(lending.head())
# print(lending.info())
# print(lending.isnull().sum().sort_values(ascending=False))
# print(lending.duplicated().sum())

# Keep useful columns for analysis
columns_to_keep = ["emp_title", "emp_length", "state", "homeownership", "annual_income", "verified_income",
                   "debt_to_income", "delinq_2y", "inquiries_last_12m", "total_credit_lines", "open_credit_lines",
                   "total_credit_limit", "total_credit_utilized", "loan_purpose", "application_type", "loan_amount",
                   "term", "interest_rate", "installment", "grade", "sub_grade", "issue_month", "loan_status",
                   "balance", "paid_total", "paid_principal", "paid_interest", "paid_late_fees"]
lending = lending[columns_to_keep].copy()

# Clean column names
lending.columns = (lending.columns.str.strip().str.lower().str.replace(" ", "_"))

# Fill missing job titles with "unknown"
lending["emp_title"] = lending["emp_title"].fillna("unknown")

# Fill missing emp_length with median
lending["emp_length"] = lending["emp_length"].fillna(lending["emp_length"].median())

# Fill missing debt_to_income with median
lending["debt_to_income"] = lending["debt_to_income"].fillna(lending["debt_to_income"].median())

# Final checks
print(lending.head())
print(lending.info())
print(lending.isnull().sum().sort_values(ascending=False))
print("Duplicate rows:", lending.duplicated().sum())

# Save cleaned file
lending.to_csv("../cleaned_data/lending_club_cleaned.csv", index=False)


# ----- API: CFPB  -----
# CFPB API URL
api_url = (
    "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
    "?date_received_min=2025-01-01"
    "&date_received_max=2025-12-31"
    "&product=Mortgage"
    "&field=all"
    "&format=csv"
    "&no_aggs=true"
    "&size=5000"
    "&sort=created_date_desc"
)

# Request raw API data
response = requests.get(api_url)
print("Status code:", response.status_code)
print(response.text[:1500])

# Save raw CSV
with open("../raw_data/cfpb_raw.csv", "w", encoding="utf-8") as f:
    f.write(response.text)

# Load into pandas
cfpb = pd.read_csv(StringIO(response.text))

# Keep useful columns
columns_to_keep = ["Date received", "Product", "Sub-product", "Issue", "Sub-issue", "Company", "State",
                   "Submitted via", "Company response to consumer", "Timely response?", "Complaint ID"]
cfpb = cfpb[columns_to_keep].copy()

# Clean column names
cfpb.columns = (cfpb.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("?", "", regex=False)
                .str.replace("-", "_"))

# Clean text fields
text_cols = ["product", "sub_product", "issue", "sub_issue", "company", "state", "submitted_via",
             "company_response_to_consumer", "timely_response"]
for col in text_cols:
    cfpb[col] = cfpb[col].astype("string").str.strip().str.lower()

# Fill missing values
cfpb["sub_product"] = cfpb["sub_product"].fillna("unknown")
cfpb["sub_issue"] = cfpb["sub_issue"].fillna("unknown")
cfpb["company_response_to_consumer"] = cfpb["company_response_to_consumer"].fillna("unknown")
cfpb["state"] = cfpb["state"].fillna("unknown")
cfpb["submitted_via"] = cfpb["submitted_via"].fillna("unknown")
cfpb["timely_response"] = cfpb["timely_response"].fillna("unknown")

# Convert date
cfpb["date_received"] = pd.to_datetime(cfpb["date_received"], errors="coerce")

# Remove duplicates
cfpb = cfpb.drop_duplicates()

# Final checks
print(cfpb.head(10))
print(cfpb.info())
print(cfpb.isnull().sum().sort_values(ascending=False))
print("Duplicate rows:", cfpb.duplicated().sum())

# Save cleaned dataset
cfpb.to_csv("../cleaned_data/cfpb_cleaned.csv", index=False)
