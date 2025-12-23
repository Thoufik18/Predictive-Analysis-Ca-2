import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



file_path = r"C:\Users\thouf\Downloads\car_price_prediction_ (1).csv"
df = pd.read_csv(file_path)

print("FIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

print("\nDATASET INFO")
print(df.info())

print("\nSTATISTICAL SUMMARY")
print(df.describe())

print("\nMISSING VALUES")
print(df.isna().sum())


df = df.dropna()

print("\nShape after dropping missing values")
print(df.shape)


sns.set(style="whitegrid")

plt.figure(figsize=(8,4))
sns.histplot(df["Price"], kde=True)
plt.title("Distribution of Car Prices")
plt.show()

plt.figure(figsize=(8,4))
sns.scatterplot(x="Mileage", y="Price", data=df)
plt.title("Price vs Mileage")
plt.show()

plt.figure(figsize=(8,4))
sns.scatterplot(x="Year", y="Price", data=df)
plt.title("Price vs Year")
plt.show()

plt.figure(figsize=(8,4))
sns.scatterplot(x="Engine Size", y="Price", data=df)
plt.title("Price vs Engine Size")
plt.show()

numeric_cols = df.select_dtypes(include="number")
plt.figure(figsize=(8,6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()



target_column = "Price"

X = df.drop(columns=[target_column, "Car ID"])
y = df[target_column]

X = pd.get_dummies(X, drop_first=True)



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)

print("\nLINEAR REGRESSION")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("MSE:", mean_squared_error(y_test, lr_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_pred)))
print("R2 Score:", r2_score(y_test, lr_pred))


dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("\nDECISION TREE REGRESSION")
print("MAE:", mean_absolute_error(y_test, dt_pred))
print("MSE:", mean_squared_error(y_test, dt_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, dt_pred)))
print("R2 Score:", r2_score(y_test, dt_pred))



knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

knn_pred = knn_model.predict(X_test_scaled)

print("\nKNN REGRESSION")
print("MAE:", mean_absolute_error(y_test, knn_pred))
print("MSE:", mean_squared_error(y_test, knn_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, knn_pred)))
print("R2 Score:", r2_score(y_test, knn_pred))



rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRANDOM FOREST REGRESSION")
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("MSE:", mean_squared_error(y_test, rf_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))
print("R2 Score:", r2_score(y_test, rf_pred))
