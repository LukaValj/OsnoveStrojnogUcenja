import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("data_C02_emission.csv")

#a)
features = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)']
target = 'CO2 Emissions (g/km)'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#b)
plt.scatter(X_train['Fuel Consumption Comb (L/100km)'], y_train, color='blue', label='Train')
plt.scatter(X_test['Fuel Consumption Comb (L/100km)'], y_test, color='red', label='Test')
plt.xlabel('Fuel Consumption Comb (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.title('CO2 Emissions vs Fuel Consumption Comb (L/100km)')
plt.legend()
plt.show()


#c)
scaler = MinMaxScaler()

plt.figure()
plt.hist(X_train['Fuel Consumption Comb (L/100km)'], bins=20, color='gray')
plt.title('Before Scaling: Fuel Consumption Comb (L/100km)')
plt.xlabel('Fuel Consumption Comb (L/100km)')
plt.ylabel('Frequency')
plt.show()

X_train_scaled = scaler.fit_transform(X_train[['Fuel Consumption Comb (L/100km)']])

X_test_scaled = scaler.transform(X_test[['Fuel Consumption Comb (L/100km)']])

X_train_scaled = pd.DataFrame(X_train_scaled, columns=['Fuel Consumption Comb (L/100km)'])
X_test_scaled = pd.DataFrame(X_test_scaled, columns=['Fuel Consumption Comb (L/100km)'])

plt.figure()
plt.hist(X_train_scaled['Fuel Consumption Comb (L/100km)'], bins=20, color='pink')
plt.title('After Scaling: Fuel Consumption Comb (L/100km)')
plt.xlabel('Scaled Fuel Consumption Comb (L/100km)')
plt.ylabel('Frequency')
plt.show()

#d)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Intercept (θ₀):", model.intercept_)
print("Koeficijenti (θ):", model.coef_)
print("\n")

#e)
y_pred = model.predict(X_test_scaled)

plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predviđene vs Stvarne')
plt.xlabel('Stvarne vrijednosti CO2 Emissions (g/km)')
plt.ylabel('Predviđene vrijednosti CO2 Emissions (g/km)')
plt.title('Stvarne vs Predviđene CO2 Emissions')
plt.legend()
plt.show()

#f)
mae = mean_absolute_error(y_test, y_pred)  # Srednja apsolutna pogreška
mse = mean_squared_error(y_test, y_pred)  # Srednja kvadratna pogreška
rmse = np.sqrt(mse)  # Korijen srednje kvadratne pogreške
r2 = r2_score(y_test, y_pred)  # Koeficijent determinacije (R²)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Srednja apsolutna postotna pogreška

print("Srednja apsolutna pogreška (MAE):", mae)
print("Srednja kvadratna pogreška (MSE):", mse)
print("Korijen srednje kvadratne pogreške (RMSE):", rmse)
print("Koeficijent determinacije (R²):", r2)
print("Srednja apsolutna postotna pogreška (MAPE):", mape)
print("\n")

#g)
for i, feat in enumerate(features, 1):
    X = df[feat] 
    if isinstance(X, pd.Series): 
        X = X.to_frame()
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 

    print(f"Nove izračunate pogreške: MAE={mae:.2f}, MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%")
    print("\n")