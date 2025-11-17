




import streamlit as st
import pandas as pd
import joblib


# تحميل النموذج المحفوظ
pipeline = joblib.load("GB.pkl")

st.set_page_config(page_title="User Conversion Prediction", layout="centered")

st.title("📈 User Conversion Prediction")
st.markdown("""
Welcome to the conversion prediction tool.  
Choose how you want to input the data: upload a CSV file or enter manually.
""")
st.markdown("---")

input_method = st.radio("Choose input method:", ["Upload CSV", "Manual Input"])

if input_method == "Upload CSV":
    st.subheader("📤 Upload CSV File")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        st.write("### Uploaded Data Preview:")
        st.dataframe(data)

    
        if st.button("Predict Conversion"):
            # Apply preprocessing pipeline to new data
            st.write("### Processing Data...")
            predictions = pipeline.predict(data)  # Apply the pipeline to new data

            data['conversion'] = predictions
            st.write("### Prediction Results:")
            st.dataframe(data)

      
            csv = data.to_csv(index=False)
            st.download_button("📥 Download Predictions as CSV", csv, file_name="predictions.csv")
else:
    # Manual input form with sliders
    st.subheader("✍️ Manual Input Features")
    col1, col2 = st.columns(2)

    with col1:
        income = st.slider("Income", min_value=0.0, max_value=100000.0, value=50000.0, step=100.0, format="%.2f")
        campaign_type = st.selectbox("Campaign Type", options=[0, 1, 2])
        ad_spend = st.slider("Ad Spend", min_value=0.0, max_value=10000.0, value=500.0, step=10.0, format="%.2f")
        ctr = st.slider("Click Through Rate (CTR)", 0.0, 1.0, value=0.05, step=0.01)
        conversion_rate = st.slider("Conversion Rate", 0.0, 1.0, value=0.02, step=0.01)
        website_visits = st.slider("Website Visits", min_value=0, max_value=1000, value=100, step=1)

    with col2:
        pages_per_visit = st.slider("Pages Per Visit", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
        time_on_site = st.slider("Time on Site (seconds)", min_value=0.0, max_value=3600.0, value=300.0, step=10.0)
        email_opens = st.slider("Email Opens", min_value=0, max_value=50, value=5, step=1)
        email_clicks = st.slider("Email Clicks", min_value=0, max_value=20, value=1, step=1)
        previous_purchases = st.slider("Previous Purchases", min_value=0, max_value=100, value=10, step=1)
        loyalty_points = st.slider("Loyalty Points", min_value=0, max_value=10000, value=500, step=10)

    if st.button("Predict Conversion"):
        user_input = pd.DataFrame({
            'Income': [income],
            'CampaignType': [campaign_type],
            'AdSpend': [ad_spend],
            'ClickThroughRate': [ctr],
            'ConversionRate': [conversion_rate],
            'WebsiteVisits': [website_visits],
            'PagesPerVisit': [pages_per_visit],
            'TimeOnSite': [time_on_site],
            'EmailOpens': [email_opens],
            'EmailClicks': [email_clicks],
            'PreviousPurchases': [previous_purchases],
            'LoyaltyPoints': [loyalty_points],
        })

        probability = pipeline.predict_proba(user_input)[0, 1]
        prediction = pipeline.predict(user_input)[0]
        
        st.markdown("### Prediction Result:")
        if prediction == 1:
         st.success(f"✔️ The model predicts that the user **WILL convert**. Probability: {probability * 100:.2f}%")
        else:
          st.warning(f"❌ The model predicts that the user **will NOT convert**. Probability: {probability * 100:.2f}%")



# streamlit run streamlit_app.py