import joblib
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from google import genai



st.set_page_config(page_title="Insurance Response Predictor", page_icon="🚗", layout="wide")

client=genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
model = joblib.load('INSURANCE_PREDICTION_model.pkl')
columns=joblib.load('model_columns.pkl')
scaler = joblib.load('SCALER1.pkl')

st.title("🚗 Insurance Customer Response Prediction")
st.caption("Predict whether a customer will purchase vehicle insurance")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Details")
    gender = st.selectbox('Gender', ['Male', 'Female'])
    gender = 1 if gender == 'Male' else 0

    age = st.slider('Age', 18, 100, 30)

    driving_license = st.selectbox('Driving License', ['Yes', 'No'])
    driving_license = 1 if driving_license == 'Yes' else 0

    region = st.selectbox('Region Code', list(range(0, 53)))

    insured = st.selectbox('Previously Insured', ['Yes', 'No'])
    insured = 1 if insured == 'Yes' else 0

with col2:
    st.subheader("🚘 Vehicle & Policy Details")
    vehicle_age = st.selectbox('Vehicle Age', ['<1 year', '1-2 year', '>2 year'])
    vehicle_age = {'<1 year': 0, '1-2 year': 1, '>2 year': 2}[vehicle_age]

    damage = st.selectbox('Vehicle Damage', ['Yes', 'No'])
    damage = 1 if damage == 'Yes' else 0

    premium = st.slider('Annual Premium', 2630, 550000, 30000, step=500)

    channel_options = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,73,74,75,76,78,79,80,81,82,83,84,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,163]
    channel = st.selectbox('Policy Sales Channel', channel_options, index=channel_options.index(152))

    vintage = st.slider('Vintage (days)', 10, 299, 100)

st.divider()

if st.button('🔮 Predict', use_container_width=True):

   model_columns = joblib.load('model_columns.pkl')
   input_data = {
      'Gender': gender,
      'Age': age,
      'Driving_License': driving_license,
      'Region_Code': region,
      'Previously_Insured': insured,
      'Vehicle_Age': vehicle_age,
      'Vehicle_Damage': damage,
      'Annual_Premium': premium,
      'Policy_Sales_Channel': channel,
      'Vintage': vintage}
      
   df_input = pd.DataFrame([input_data])
   prob = model.predict_proba(df_input)[0]
   best_threshold=0.6628225768
   pred=int(prob[1]>=best_threshold)

   result_col, gauge_col = st.columns(2)

   if pred == 1:
          st.success("✅ Customer is LIKELY to purchase insurance")
          result_text='WILL PURCHASE'
   else:
          st.error("❌ Customer is UNLIKELY to purchase insurance")
          result_text='WILL NOT PURCHASE'
   m1, m2 = st.columns(2)
   m1.metric("Purchase Probability", f"{prob[1]*100:.1f}%")
   m2.metric("Non-Purchase Probability", f"{prob[0]*100:.1f}%") 
   st.divider()
   st.subheader('Prediction Breakdown')
   imp_col,guage_col=st.columns(2)
   with imp_col:

     if hasattr(model, "feature_importances_"):
         st.subheader("📊 What Drove This Prediction")
         importances = model.feature_importances_
         imp_df = pd.DataFrame({
                    'Feature': model_columns,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)

         fig_imp = go.Figure(go.Bar(
                    x=imp_df['Importance'],
                    y=imp_df['Feature'],
                    orientation='h',
                    marker_color='#3498db'
                ))
         fig_imp.update_layout(height=350,template='plotly_dark', margin=dict(t=10, b=0,l=10,r=10))
         st.plotly_chart(fig_imp, use_container_width=True)
            



   with gauge_col:
         st.caption('Likelihood to Buy')
         fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob[1]*100,
                title={'text': "Likelihood to Buy (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2ecc71" if prob[1] >= best_threshold else "#e74c3c"},
                    'steps': [
                    {'range': [0, 40], 'color': "#fdecea"},
                    {'range': [40, 70], 'color': "#fff8e1"},
                    {'range': [70, 100], 'color': "#e8f5e9"}
                ]
            }
        ))
         fig.update_layout(height=300,template='plotly_dark', margin=dict(t=60, b=20,l=30,r=30))
         st.plotly_chart(fig, use_container_width=True)

   with st.spinner("Generating AI insights..."):
        prompt = (
            f"A machine learning model predicted that a customer with Age {age},"
            f" Vehicle Age {vehicle_age}, Vehicle Damage"
            f" {damage}, and Annual Premium {premium}"
            f" {result_text} vehicle insurance. Provide a brief, professional"
            " 2-sentence explanation for why this outcome makes sense based on"
            " standard insurance risk factors."
        )

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
            )
            ai_explanation = response.text
        except Exception as e:
            ai_explanation = f"⚠️ Could not generate AI insights: {e}"

        st.subheader("🤖 AI Risk Insights")
        st.write(ai_explanation)

  
        
