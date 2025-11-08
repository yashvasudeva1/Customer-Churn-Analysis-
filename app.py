import streamlit as st
st.title("Customer Churn Analysis")
st.write("This project analyzes customer data to understand and predict churn — identifying customers who are likely to discontinue a service. By uncovering key behavioral and demographic factors behind churn, this analysis helps businesses make data-driven retention decisions.")
tab1,tab2,tab3,tab4,tab5=st.tabs(["Overview","About Data","Visualisation","Findings","Prediction"])
import joblib
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
 st.subheader("Basic Structure")
 st.write(data_structure_d)
with tab3:
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) > 0:
        selected_cols = st.multiselect("Select numeric columns to plot:", numeric_cols)
    
        if selected_cols:
            st.line_chart(df[selected_cols])
        else:
            st.info("Please select at least one column to plot.")
    else:
        st.warning("No numeric columns found in the dataset.")
    st.image('download (1).png')
    st.image('download (2).png')
    st.image('download.png')
    st.image('train vs test.png')

with tab4:
    st.markdown("### Main Findings and Insights")
    
    st.markdown("#### 1. Churners usually have higher MonthlyCharges")
    st.write("""
    When I compared the monthly charges of customers who stayed versus those who left, 
    I found that churners tend to pay noticeably higher monthly charges.  
    The Mann–Whitney U test produced a p-value of approximately 3.31e-54, 
    which is extremely small. This means the difference is statistically significant, 
    and I can confidently conclude that customers who churn generally pay more each month.
    """)
    
    st.markdown("#### 2. Customers who churn generally have shorter tenures")
    st.write("""
    I also examined how long customers have been with the company.  
    The test results (p ≈ 2.42e-208) clearly indicate that customers who churn 
    are usually newer and have much shorter tenures.  
    In simple terms, people who have recently joined are more likely to leave early.
    """)
    
    st.markdown("#### 3. Contract type, payment method, and internet service are strongly linked to churn")
    st.write("""
    When I ran Chi-square tests on categorical variables like contract type, payment method, 
    and internet service, the p-values were all extremely small (much less than 0.001).  
    This shows a strong statistical association with churn.
    
    Here’s what I observed:
    - Customers on month-to-month contracts are much more likely to churn.  
    - Those paying through electronic checks have higher churn rates.  
    - Customers without long-term contracts (such as one- or two-year plans) tend to leave more often.
    """)
    
    st.markdown("#### 4. Numeric relationships and correlations")
    st.write("""
    There is a clear positive relationship between tenure and TotalCharges, 
    which makes sense since customers who stay longer pay more in total.  
    MonthlyCharges also show a positive correlation with churn, 
    indicating that higher-paying customers are slightly more likely to leave.  
    These findings were confirmed through correlation matrices and visual analysis.
    """)
    
    st.markdown("#### 5. Data quality and missing values")
    st.write("""
    During analysis, I noticed a few missing values in the TotalCharges column.  
    After reviewing, I found that these correspond to customers with tenure equal to zero, 
    likely those who just joined.  
    It would be reasonable to either impute these missing values as zero or remove those rows entirely before modeling.
    """)
    
    st.markdown("#### 6. Observations from distributions and visuals")
    st.write("""
    The distribution of tenure is heavily skewed toward lower values, 
    meaning there are many new customers in the dataset.  
    MonthlyCharges is right-skewed, with most customers paying lower amounts 
    and a few paying significantly more.  
    Boxplots show a clear difference in median MonthlyCharges between churned and retained customers, 
    with churned customers generally paying higher amounts.
    """)
    
    st.divider()
    
    st.markdown("## Practical Conclusions and Recommendations")
    
    st.markdown("#### High-Risk Churn Profile")
    st.write("""
    Based on the findings, customers most likely to churn are those who:
    - Have month-to-month contracts,  
    - Are new or have low tenure,  
    - Pay higher monthly charges, or  
    - Use electronic check as their payment method.
    """)
    
    st.markdown("#### Recommended Business Actions")
    st.write("""
    If I were to propose next steps for the business, I would recommend:
    - Encouraging month-to-month customers to switch to one- or two-year contracts through discounts or bundled offers.  
    - Engaging newly joined customers with onboarding campaigns and retention programs.  
    - Reviewing high-paying customers and providing loyalty benefits or personalized offers.  
    - Investigating payment issues or friction among customers using electronic checks.
    """)

with tab5:
    import streamlit as st
    import pandas as pd
    import joblib
    
    st.set_page_config(page_title="Telco Customer Churn Prediction", layout="wide")
    
    st.title("Telco Customer Churn Prediction App")
    
    @st.cache_resource
    def load_model():
        model = joblib.load("churn_pipeline.pkl")
        return model
    
    model = load_model()
    
    st.subheader("Enter Customer Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
    
    with col2:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    
    with col3:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1400.0)
    
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "InternetService": internet_service,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])
    
    st.write("### Input Summary")
    st.dataframe(input_data)
    
    if st.button("Predict Churn"):
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
    
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"The customer is likely to CHURN (probability = {probability:.2f})")
            else:
                st.success(f"The customer is likely to STAY (probability = {probability:.2f})")
    
            st.write(f"Model confidence: {probability*100:.2f}%")
        except Exception as e:
            st.error(f"Error during prediction: {e}")    

