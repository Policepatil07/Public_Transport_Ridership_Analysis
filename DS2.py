import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import streamlit as st

df=pd.read_csv("C:/Users/INTEL-PC/Downloads/public_transport_data.csv")

df.info()
df.isnull().sum()
df.describe()
print(df.head())

df['operating_day'] = pd.to_datetime(df['operating_day'])
df['arrival'] = pd.to_datetime(df['arrival'])
df['departure'] = pd.to_datetime(df['departure'])

df = df.dropna()

df['year'] = df['operating_day'].dt.year
df['month'] = df['operating_day'].dt.month
df['day_name'] = df['operating_day'].dt.day_name()
df['week_of_year'] = df['operating_day'].dt.isocalendar().week

total_passengers = df['passengers'].sum()
print("Total Passengers:", total_passengers)

avg_load_factor = df['load_factor'].mean()
print("Average Load Factor:", avg_load_factor)

#EDA********************************************************

#Daily Passenger Trend Analysis

daily_passengers = df.groupby('operating_day')['passengers'].sum()

plt.figure(figsize=(10,6))
daily_passengers.plot(color='blue')
plt.title("Daily Passenger Trend")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.grid(True)
plt.show()

#Monthly Passenger Analysis

monthly_passengers = df.groupby('month')['passengers'].sum()

plt.figure(figsize=(10,5))
sns.barplot(x=monthly_passengers.index,y=monthly_passengers.values,palette='viridis')
plt.title("Monthly Passenger Analysis")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.show()

#Weekday Analysis

weekday_analysis = df.groupby('day_of_week')['passengers'].sum()

plt.figure(figsize=(10,5))
weekday_analysis.plot(kind='bar',color='orange')
plt.title("Weekday Passenger Analysis")
plt.xlabel("Day")
plt.ylabel("Passengers")
plt.show()

#Peak Hour Analysis

peak_hour_data = df.groupby('departure_hour')['passengers'].sum()

plt.figure(figsize=(12,5))
sns.lineplot(x=peak_hour_data.index,y=peak_hour_data.values,marker='o')
plt.title("Peak Hour Passenger Analysis")
plt.xlabel("Hour")
plt.ylabel("Passengers")
plt.show()

#Route-wise Passenger Analysis

route_analysis = df.groupby('line_id')['passengers'].sum()
top_routes = route_analysis.sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))
top_routes.plot(kind='bar',color='green')
plt.title("Top 10 Routes by Passengers")
plt.xlabel("Route ID")
plt.ylabel("Passengers")
plt.show()

#Stop-wise Passenger Analysis

stop_analysis = df.groupby('stop_id')['passengers'].sum()
top_stops = stop_analysis.sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))
top_stops.plot(kind='bar',color='purple')
plt.title("Top 10 Stops by Passengers")
plt.xlabel("Stop ID")
plt.ylabel("Passengers")
plt.show()

#Weekend vs Weekday Analysis

weekend_analysis = df.groupby('is_weekend')['passengers'].sum()

plt.figure(figsize=(6,5))
sns.barplot(x=weekend_analysis.index,y=weekend_analysis.values,palette='Set2')
plt.title("Weekend vs Weekday Passengers")
plt.xlabel("Is Weekend")
plt.ylabel("Passengers")
plt.show()

#Part of Day Analysis

part_day_analysis = df.groupby('part_of_day')['passengers'].sum()

plt.figure(figsize=(8,5))
part_day_analysis.plot(kind='pie',autopct='%1.1f%%')
plt.title("Passenger Distribution by Part of Day")
plt.ylabel("")
plt.show()

#Load Factor Analysis

plt.figure(figsize=(10,5))
sns.histplot(df['load_factor'],bins=30,kde=True,color='red')
plt.title("Load Factor Distribution")
plt.xlabel("Load Factor")
plt.show()

#Heatmap Analysis

heatmap_data = df.pivot_table(values='passengers',index='day_of_week',columns='departure_hour',aggfunc='sum')

plt.figure(figsize=(14,6))
sns.heatmap(heatmap_data,cmap='YlGnBu')
plt.title("Passenger Heatmap")
plt.show()

#Correlation Analysis

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(14,8))
sns.heatmap(numeric_df.corr(),annot=True,cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

#Machine Learning Model Feature Selection

features = ['month','departure_hour','trip_stop_sum','vehicle_seats','load_factor']

X = df[features]
y = df['passengers']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

#Actual vs Predicted Visualization

plt.figure(figsize=(10,6))

plt.scatter(y_test,predictions,alpha=0.5,color='blue')
plt.xlabel("Actual Passengers")
plt.ylabel("Predicted Passengers")
plt.title("Actual vs Predicted Passengers")
plt.show()

importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(8,5))

sns.barplot(x='Importance',y='Feature',data=importance,palette='viridis')

plt.title("Feature Importance")
plt.show()