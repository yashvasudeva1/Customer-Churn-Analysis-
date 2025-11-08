import streamlit as st
st.title("Customer Churn Analysis")
st.write("This project analyzes customer data to understand and predict churn — identifying customers who are likely to discontinue a service. By uncovering key behavioral and demographic factors behind churn, this analysis helps businesses make data-driven retention decisions.")
tab1,tab2,tab3,tab4=st.tabs(["Overview","About Data","Visualisation","Prediction"])

import numpy as np
import pandas as pd
df=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
with tab1:
  st.write(df)
  st.subheading("Basic Statistics")
