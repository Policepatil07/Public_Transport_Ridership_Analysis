import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import streamlit as st

df=pd.read_csv("C:/Users/INTEL-PC/Downloads/public_transport_data.csv")

df['operating_day'] = pd.to_datetime(df['operating_day'])
df['day_name'] = df['operating_day'].dt.day_name()

st.title("Public Transport Passenger Analysis & Prediction 🚍")

st.sidebar.header("Filters")

selected_month = st.sidebar.slider("Select Month", 1, 12, (1,12))
selected_hour = st.sidebar.slider("Select Departure Hour", 0, 23, (0,23))
selected_weekend = st.sidebar.selectbox("Weekend or Weekday?", ["All", "Weekday", "Weekend"])

filtered_df = df[
    (df['month'] >= selected_month[0]) & (df['month'] <= selected_month[1]) &
    (df['departure_hour'] >= selected_hour[0]) & (df['departure_hour'] <= selected_hour[1])]

if selected_weekend == "Weekday":
    filtered_df = filtered_df[filtered_df['is_weekend'] == 0]
elif selected_weekend == "Weekend":
    filtered_df = filtered_df[filtered_df['is_weekend'] == 1]

st.subheader("Filtered Data Preview")
st.dataframe(filtered_df.head(10))

# Passenger Trends 
st.subheader("Daily Passenger Trend")
daily_passengers = filtered_df.groupby('operating_day')['passengers'].sum()
st.line_chart(daily_passengers)

st.subheader("Monthly Passenger Analysis")
monthly_passengers = filtered_df.groupby('month')['passengers'].sum()
st.bar_chart(monthly_passengers)

st.subheader("Weekday Passenger Analysis")
weekday_passengers = filtered_df.groupby('day_name')['passengers'].sum()
st.bar_chart(weekday_passengers)

st.subheader("Peak Hour Passenger Analysis")
hour_passengers = filtered_df.groupby('departure_hour')['passengers'].sum()
st.line_chart(hour_passengers)

st.subheader("Top 10 Routes by Passengers")
top_routes = filtered_df.groupby('line_id')['passengers'].sum().sort_values(ascending=False).head(10)
st.bar_chart(top_routes)

st.subheader("Top 10 Stops by Passengers")
top_stops = filtered_df.groupby('stop_id')['passengers'].sum().sort_values(ascending=False).head(10)
st.bar_chart(top_stops)

# Load Factor 
st.subheader("Load Factor Distribution")
fig, ax = plt.subplots()
sns.histplot(filtered_df['load_factor'], bins=30, kde=True, color='red', ax=ax)
st.pyplot(fig)

# Heatmap
st.subheader("Passenger Heatmap: Day vs Hour")
heatmap_data = filtered_df.pivot_table(values='passengers', index='day_name', columns='departure_hour', aggfunc='sum', fill_value=0)
fig, ax = plt.subplots(figsize=(12,6))
sns.heatmap(heatmap_data, cmap='YlGnBu', ax=ax)
st.pyplot(fig)

# Predict Passengers 
st.subheader("Predict Passenger Count")
features = ['month','departure_hour','trip_stop_sum','vehicle_seats','load_factor']

X = df[features]
y = df['passengers']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# User Inputs
st.markdown("### Enter Feature Values:")
month_input = st.slider("Month", 1, 12, 6)
hour_input = st.slider("Departure Hour", 0, 23, 9)
trip_stop_input = st.number_input("Trip Stop Sum", min_value=1, max_value=100, value=10)
vehicle_seats_input = st.number_input("Vehicle Seats", min_value=10, max_value=200, value=50)
load_factor_input = st.slider("Load Factor", 0.0, 1.0, 0.5)

input_features = np.array([[month_input, hour_input, trip_stop_input, vehicle_seats_input, load_factor_input]])
predicted_passengers = model.predict(input_features)[0]

st.success(f"Predicted Passengers: {int(predicted_passengers)}")

# Model Performance
st.subheader("Model Performance on Test Set")
preds = model.predict(X_test)
st.write(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")
st.write(f"R2 Score: {r2_score(y_test, preds):.2f}")

# Feature Importance 
st.subheader("Feature Importance")
importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
st.bar_chart(importance.set_index('Feature'))