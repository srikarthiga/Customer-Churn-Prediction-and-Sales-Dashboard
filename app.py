import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Data Setup (Replace with your real dataset)
@st.cache_data
def load_data():
    data = {
        'customer_id': [1,2,3,4,5,6,7,8,9,10],
        'date': pd.date_range(start='2023-01-01', periods=10, freq='M'),
        'amount': [100, 150, 200, 50, 300, 250, 400, 350, 100, 80],
        'churn_flag': [0,0,1,0,1,0,0,1,0,0],
        'segment': ['A', 'B', 'A', 'A', 'C', 'B', 'C', 'C', 'B', 'A']
    }
    return pd.DataFrame(data)

df = load_data()
df['month'] = df['date'].dt.to_period('M')

st.title("Interactive Customer Retention & Sales Dashboard")

# 1. Real-time Customer Retention and Churn Analysis
st.header("Customer Retention & Churn Analysis")

churn_rate = df['churn_flag'].mean() * 100
st.metric(label="Overall Churn Rate (%)", value=f"{churn_rate:.2f}")

monthly_churn = df.groupby('month')['churn_flag'].mean() * 100
st.line_chart(monthly_churn.to_timestamp())

# 2. Sales Visualization
st.header("Sales Visualization")

monthly_sales = df.groupby('month')['amount'].sum()
st.line_chart(monthly_sales.to_timestamp())

# Heatmap of sales by segment and month
sales_pivot = df.pivot_table(index='segment', columns='month', values='amount', aggfunc='sum').fillna(0)
fig, ax = plt.subplots(figsize=(8,4))
sns.heatmap(sales_pivot, annot=True, fmt=".0f", cmap='Blues', ax=ax)
st.pyplot(fig)

# 3. Customer Segmentation Insights
st.header("Customer Segmentation Insights")

segment_counts = df['segment'].value_counts()
st.bar_chart(segment_counts)

segment_churn = df.groupby('segment')['churn_flag'].mean() * 100
st.write("Churn Rate by Segment (%)")
st.dataframe(segment_churn)

# Optionally, allow filtering by segment
selected_segment = st.selectbox("Select Segment", options=df['segment'].unique())
filtered_df = df[df['segment'] == selected_segment]

st.write(f"Details for segment: {selected_segment}")
st.dataframe(filtered_df)
