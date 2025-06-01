import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Crop Production Prediction", layout="wide")

st.title("🌾 Crop Production Prediction")
st.markdown("Predict total crop production (in tons) using area harvested, yield, and year.")

# Corrected file path using raw string (r"") or forward slashes (/)
df = pd.read_excel(r"c:\Users\ud640\OneDrive\Desktop\GUVI PROJECTS\FAOSTAT_data.xlsx")

# Clean and filter relevant data
df = df[df['Element'].isin(['Area harvested', 'Yield', 'Production'])]

# Pivot to get columns for Area, Yield, and Production
df_pivot = df.pivot_table(index=['Area', 'Item', 'Year'], columns='Element', values='Value').reset_index()

# Drop rows with missing values
df_pivot = df_pivot.dropna()

# Rename columns for simplicity
df_pivot.columns.name = None
df_pivot.rename(columns={
    'Area harvested': 'Area_harvested',
    'Production': 'Production',
    'Yield': 'Yield'
}, inplace=True)

# Sidebar - User input
st.sidebar.header("🔧 Select Prediction Inputs")
area = st.sidebar.selectbox("Region", df_pivot['Area'].unique())
crop = st.sidebar.selectbox("Crop Type", df_pivot['Item'].unique())
year = st.sidebar.selectbox("Year", sorted(df_pivot['Year'].unique()))
area_harvested = st.sidebar.number_input("Area Harvested (ha)", min_value=0.0)
yield_input = st.sidebar.number_input("Yield (kg/ha)", min_value=0.0)

# Filter dataset
filtered_df = df_pivot[(df_pivot['Area'] == area) & (df_pivot['Item'] == crop)]

# Features and Target
X = filtered_df[['Year', 'Area_harvested', 'Yield']]
y = filtered_df['Production']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predict based on user input
input_data = np.array([[year, area_harvested, yield_input]])

lr_pred = lr.predict(input_data)[0]
rf_pred = rf.predict(input_data)[0]

st.subheader("📈 Predicted Crop Production (in tons)")
st.write(f"*Linear Regression Prediction:* {lr_pred:,.2f} tons")
st.write(f"*Random Forest Prediction:* {rf_pred:,.2f} tons")

# Evaluation
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

def show_metrics(name, y_true, y_pred):
    st.markdown(f"{name} Metrics**")
    st.write("R² Score:", round(r2_score(y_true, y_pred), 3))
    st.write("MAE:", round(mean_absolute_error(y_true, y_pred), 2))
    st.write("MSE:", round(mean_squared_error(y_true, y_pred), 2))

st.subheader("📊 Model Evaluation")
col1, col2 = st.columns(2)
with col1:
    show_metrics("Linear Regression", y_test, y_pred_lr)
with col2:
    show_metrics("Random Forest", y_test, y_pred_rf)

# EDA
st.subheader("🔍 Exploratory Data Analysis")
tab1, tab2, tab3 = st.tabs(["Yield vs Production", "Production by Year", "Heatmap"])

with tab1:
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df, x='Yield', y='Production', hue='Year', ax=ax)
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots()
    sns.lineplot(data=filtered_df, x='Year', y='Production', marker='o', ax=ax)
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots()
    sns.heatmap(df_pivot[['Year', 'Area_harvested', 'Yield', 'Production']].corr(), annot=True, cmap='YlGnBu', ax=ax)
    st.pyplot(fig)
