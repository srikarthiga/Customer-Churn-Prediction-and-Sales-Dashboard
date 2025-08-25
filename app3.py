# =============================
# app.py - Interactive Dashboard + ML Models with Graphs
# =============================

# ===== Streamlit & Viz Libraries =====
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Supervised Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

# Unsupervised Learning
from sklearn.cluster import KMeans, DBSCAN


# =============================
# 1️⃣ LOAD SAMPLE DATA
# =============================
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

st.title("📊 Interactive Customer Retention & Sales Dashboard")

# =============================
# 2️⃣ CHURN ANALYSIS
# =============================
st.header("Customer Retention & Churn Analysis")

churn_rate = df['churn_flag'].mean() * 100
st.metric(label="Overall Churn Rate (%)", value=f"{churn_rate:.2f}")

monthly_churn = df.groupby('month')['churn_flag'].mean() * 100
st.line_chart(monthly_churn.to_timestamp())

# =============================
# 3️⃣ SALES VISUALIZATION
# =============================
st.header("Sales Visualization")

monthly_sales = df.groupby('month')['amount'].sum()
st.line_chart(monthly_sales.to_timestamp())

# Heatmap
sales_pivot = df.pivot_table(index='segment', columns='month', values='amount', aggfunc='sum').fillna(0)
fig, ax = plt.subplots(figsize=(8,4))
sns.heatmap(sales_pivot, annot=True, fmt=".0f", cmap='Blues', ax=ax)
st.pyplot(fig)

# =============================
# 4️⃣ SEGMENTATION INSIGHTS
# =============================
st.header("Customer Segmentation Insights")

segment_counts = df['segment'].value_counts()
st.bar_chart(segment_counts)

segment_churn = df.groupby('segment')['churn_flag'].mean() * 100
st.write("Churn Rate by Segment (%)")
st.dataframe(segment_churn)

selected_segment = st.selectbox("Select Segment", options=df['segment'].unique())
filtered_df = df[df['segment'] == selected_segment]
st.write(f"Details for segment: {selected_segment}")
st.dataframe(filtered_df)

# =============================
# 5️⃣ EXTRA VISUALIZATION DASHBOARD
# =============================
st.header("Additional Sales Dashboard")

# Quarterly & Yearly aggregation
df['quarter'] = df['date'].dt.to_period('Q')
df['year'] = df['date'].dt.year

quarterly_sales = df.groupby('quarter')['amount'].sum()
yearly_sales = df.groupby('year')['amount'].sum()

# Dummy top products
top_products = pd.Series({'A':500,'B':400,'C':350,'D':300})

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Monthly trend
axs[0, 0].plot(monthly_sales.index.astype(str), monthly_sales.values, marker='o', color='blue')
axs[0, 0].set_title("Monthly Sales Trend")
axs[0, 0].tick_params(axis='x', rotation=45)

# Quarterly sales
axs[0, 1].bar(quarterly_sales.index.astype(str), quarterly_sales.values, color='orange')
axs[0, 1].set_title("Quarterly Sales")
axs[0, 1].tick_params(axis='x', rotation=45)

# Yearly sales
axs[1, 0].bar(yearly_sales.index.astype(str), yearly_sales.values, color='green')
axs[1, 0].set_title("Yearly Sales")

# Top products
axs[1, 1].bar(top_products.index, top_products.values, color='purple')
axs[1, 1].set_title("Top Performing Products")

plt.tight_layout()
st.pyplot(fig)

# =============================
# 6️⃣ MACHINE LEARNING SECTION
# =============================
st.header("ML Models - Churn Prediction")

df_ml = pd.DataFrame({
    'product_id': [1,2,3,4,5,6,7,8],
    'product_name': ['A','B','C','D','E','F','G','H'],
    'category': ['Electronics','Clothing','Clothing','Electronics','Furniture','Furniture','Electronics','Clothing'],
    'unit_price': [1200, 50, 40, 1100, 300, 250, 900, 60],
    'churn_flag': [1,0,0,1,0,0,1,0]
})

le = LabelEncoder()
df_ml['category'] = le.fit_transform(df_ml['category'])

X = df_ml[['category', 'unit_price']]
y = df_ml['churn_flag']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Supervised Models
log_reg = LogisticRegression().fit(X_train, y_train)
dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
xgb = XGBClassifier(eval_metric='logloss').fit(X_train, y_train)
lgb = LGBMClassifier().fit(X_train, y_train)

acc_dict = {
    "Logistic Regression": accuracy_score(y_test, log_reg.predict(X_test)),
    "Decision Tree": accuracy_score(y_test, dt.predict(X_test)),
    "Random Forest": accuracy_score(y_test, rf.predict(X_test)),
    "XGBoost": accuracy_score(y_test, xgb.predict(X_test)),
    "LightGBM": accuracy_score(y_test, lgb.predict(X_test))
}

st.subheader("📌 Supervised Model Accuracies")
st.write(acc_dict)
st.bar_chart(pd.Series(acc_dict))

# =============================
# 7️⃣ DEEP LEARNING MODELS
# =============================
st.subheader("📌 Deep Learning Models Training... (sample)")

# ANN
ann = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
ann.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, epochs=5, verbose=0)
_, acc_ann = ann.evaluate(X_test, y_test, verbose=0)

# RNN
X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

rnn = Sequential([
    LSTM(16, activation='relu', input_shape=(1, X_train.shape[1])),
    Dense(1, activation='sigmoid')
])
rnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
rnn.fit(X_train_rnn, y_train, epochs=5, verbose=0)
_, acc_rnn = rnn.evaluate(X_test_rnn, y_test, verbose=0)

deep_dict = {"ANN Accuracy": round(acc_ann, 2), "RNN Accuracy": round(acc_rnn, 2)}

st.write(deep_dict)
st.bar_chart(pd.Series(deep_dict))

# =============================
# 8️⃣ UNSUPERVISED LEARNING
# =============================
st.header("Unsupervised Learning")

kmeans = KMeans(n_clusters=2, random_state=42)
df_ml['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

dbscan = DBSCAN(eps=1.5, min_samples=2)
df_ml['Cluster_DBSCAN'] = dbscan.fit_predict(X_scaled)

st.dataframe(df_ml)
