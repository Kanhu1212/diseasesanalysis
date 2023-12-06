import numpy as np
import pandas as pd
import datetime as dt
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet import plot
from prophet.plot import plot_plotly,plot_components_plotly
import streamlit as st
import plotly.graph_objects as go

state_name=['Tripura', 'Assam', 'Meghalaya', 'Jammu And Kashmir', 'Punjab',
       'Chandigarh', 'Kerala', 'Nagaland', 'Daman And Diu',
       'Dadra And Nagar Haveli', 'Uttar Pradesh', 'Delhi', 'Haryana',
       'Uttarakhand', 'Madhya Pradesh', 'Gujarat', 'Bihar', 'Mizoram',
       'Jharkhand', 'Manipur', 'Chhattisgarh', 'Maharashtra',
       'Himachal Pradesh', 'Odisha', 'Puducherry', 'Karnataka',
       'Andaman And Nicobar Islands', 'Sikkim', 'Telangana',
       'Arunachal Pradesh', 'Rajasthan', 'West Bengal', 'Tamil Nadu',
       'Andhra Pradesh']
Duration=['2019-2023']
diseases=['ACCELERATED HYPERTENSION', 'SEVERE SEPSIS',
       'CONGESTIVE HEART FAILURE']
st.title('Diseases Analysis')
col1,col2,col3  = st.columns(3)
with col1:
    Duration = st.selectbox('Duration from 2019-2023',Duration)
with col2:
    state_name = st.selectbox('States',state_name)
with col3:
    diseases = st.selectbox('Diseases',diseases)

df = pd.read_excel('Diseases_data.xlsx')
df['month'] = df['admission_dt_pat'].dt.strftime('%m %B')
df['year'] = df['admission_dt_pat'].dt.year
st.subheader('Data Description')
st.write(df.describe())

st.subheader('Sample Data')
st.write(df.head(4))

st.header('Daily Trend Of The Data')
df_state_diseases = df[(df['hosp_state']==state_name) & (df['proc_name_1']==diseases)]
df_state_diseases = df_state_diseases.groupby('admission_dt_pat').size().reset_index(name='count')
fig = plt.figure(figsize=(25,7))
plt.plot('admission_dt_pat','count',data=df_state_diseases)
plt.xticks(fontsize=15)
plt.show()
st.pyplot(fig)

st.header('Monthly Trend Of The Data')
df_state_diseases = df[(df['hosp_state']==state_name) & (df['proc_name_1']==diseases)]
df_state_diseases = df_state_diseases.groupby('month').size().reset_index(name='count')
fig = plt.figure(figsize=(25,7))
plt.plot('month','count',data=df_state_diseases)
plt.xticks(fontsize=15)
plt.show()
st.pyplot(fig)

st.header('Yearly Trend Of The Data')
df_state_diseases = df[(df['hosp_state']==state_name) & (df['proc_name_1']==diseases)]
df_state_diseases = df_state_diseases.groupby('year').size().reset_index(name='count')
fig = plt.figure(figsize=(25,7))
plt.plot('year','count',data=df_state_diseases)
plt.xticks(fontsize=15)
plt.show()
st.pyplot(fig)

st.header('Forcasting on the Data')
m_state_name= Prophet()
df_state_diseases = df[(df['hosp_state']==state_name) & (df['proc_name_1']==diseases)]
diseases_state_name_pro = df_state_diseases.groupby('admission_dt_pat').size().reset_index(name='count')
diseases_state_name_pro.columns = ['ds','y']
pro_state_name= m_state_name.fit(diseases_state_name_pro)

futr_state_name = m_state_name.make_future_dataframe(periods=1000,freq='D')
frcst_state_name = pro_state_name.predict(futr_state_name)
st.subheader('Trend')
st.write(frcst_state_name[['ds','yhat']].tail(10))
#fig = plt.figure(figsize=(25,7))
fig = plot_plotly(pro_state_name,frcst_state_name)
st.plotly_chart(fig)

fig = plot_components_plotly(pro_state_name,frcst_state_name)
st.plotly_chart(fig)