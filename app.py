import streamlit as st
st.title("Customer Churn Analysis")
st.write("This project analyzes customer data to understand and predict churn — identifying customers who are likely to discontinue a service. By uncovering key behavioral and demographic factors behind churn, this analysis helps businesses make data-driven retention decisions.")
tab1,tab2,tab3,tab4=st.tabs(["Overview","About Data","Visualisation","Prediction"])

import numpy as np
import pandas as pd
df=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
num_rows, num_cols=df.shape
numerical_features=df.select_dtypes(include=np.number).shape[1]
categorical_features=df.select_dtypes(include='object').shape[1]
data_structure={
    'Metric': ['Number of Rows', 'Number of Columns', 'Numerical Features', 'Categorical Features'],
    'Count': [num_rows, num_cols, numerical_features, categorical_features]
}
data_structure_d=pd.DataFrame(data_structure)
with tab1:
  st.subheader("Dataset Overview")
  st.write(df)
  st.subheader("Basic Statistics")
  st.write(df.describe())

with tab2:
 st.subheader("Basic Structure")
 st.write(data_structure_d)
 st.subheader("About the Dataset")
 st.markdown("""
This dataset is sourced from **Kaggle’s Telco Customer Churn** dataset, originally created by **IBM Sample Data Sets**.

**Context:**  
The goal is to predict customer behavior and identify patterns that lead to **customer churn** — i.e., customers who stop using a service.  
By analyzing the data, businesses can develop focused **customer retention strategies**.

**Content Overview:**  
Each row represents a customer, and each column contains attributes describing their demographics, services, and account details.

- **Churn:** Whether the customer left within the last month.  
- **Services:** Includes phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming (TV & movies).  
- **Account Information:** Tenure, contract type, payment method, paperless billing, monthly charges, and total charges.  
- **Demographics:** Gender, senior citizen status, partner, and dependents.

**Inspiration:**  
To understand and predict customer churn through exploratory data analysis and machine learning, helping organizations improve customer retention.

**Source:**  
[IBM Telco Customer Churn Dataset](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)
""")
