import streamlit as st
import pandas as pd

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Churn & Sales Dashboard", layout="wide")

# ---- Load Data ----
@st.cache_data
def load_data():
    customers = pd.read_csv("customer_data.csv")
    transactions = pd.read_csv("transaction_data.csv")
    return customers, transactions

customers, transactions = load_data()

# ---- Merge Data ----
if "CustomerID" in customers.columns and "CustomerID" in transactions.columns:
    df = pd.merge(transactions, customers, on="CustomerID", how="left")
else:
    st.error("❌ Both CSV files must have 'CustomerID' column to merge.")
    st.stop()

# ---- Churn Calculation ----
if "Churn" in customers.columns:
    churn_rate = (customers["Churn"].mean()) * 100
else:
    churn_rate = None

# ---- Sales KPIs ----
if "Sales" in transactions.columns:
    total_sales = transactions["Sales"].sum()
    avg_sales = transactions["Sales"].mean()
else:
    total_sales = None
    avg_sales = None

# ---- Dashboard Layout ----
st.title("📊 Customer Churn & Sales Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    if churn_rate is not None:
        st.metric("Churn Rate", f"{churn_rate:.2f}%")
    else:
        st.warning("⚠️ 'Churn' column not found in customer_data.csv")

with col2:
    if total_sales is not None:
        st.metric("Total Sales", f"₹{total_sales:,.0f}")
    else:
        st.warning("⚠️ 'Sales' column not found in transaction_data.csv")

with col3:
    if avg_sales is not None:
        st.metric("Average Sales per Transaction", f"₹{avg_sales:,.2f}")
    else:
        st.warning("⚠️ 'Sales' column missing for Avg calculation")

# ---- Show Data ----
st.subheader("Customer Data Preview")
st.dataframe(customers.head())

st.subheader("Transaction Data Preview")
st.dataframe(transactions.head())

st.subheader("Merged Data Preview")
st.dataframe(df.head())
