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
  st.write(data_structure_d)
