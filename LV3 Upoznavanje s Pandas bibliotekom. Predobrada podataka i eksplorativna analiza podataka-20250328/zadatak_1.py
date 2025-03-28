import pandas as pd
import numpy as np
df = pd.read_csv('data_C02_emission.csv')

#a)
print(f'Broj redaka: {df.shape[0]}')
print(f'Broj stupaca: {df.shape[1]}')
print(df.dtypes)
print(df.isnull().sum())
df = df.dropna() #izbacijue ovo čega nema sve
print(df.duplicated().sum())
df = df.drop_duplicates() #brisanje dupliciranjih vrijednosti
categorical_columns = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
df[categorical_columns] = df[categorical_columns].astype('category')

#b)
df_sorted_desc = df.sort_values(by='Fuel Consumption City (L/100km)', ascending=False)
df_sorted_asc = df.sort_values(by='Fuel Consumption City (L/100km)', ascending=True)

print("Tri automobila s najvećom gradskom potrošnjom:")
for index, row in df_sorted_desc.head(3).iterrows():
    print(f'{row["Make"]} {row["Model"]} - Gradska potrošnja: {row["Fuel Consumption City (L/100km)"]} L/100km')
print("\n")
print("Tri automobila s najmanjom gradskom potrošnjom:")
for index, row in df_sorted_asc.head(3).iterrows():
    print(f'{row["Make"]} {row["Model"]} - Gradska potrošnja: {row["Fuel Consumption City (L/100km)"]} L/100km')
print("\n")

#c)
filtered_vehicles = df[(df['Engine Size (L)'] >= 2.5) & (df['Engine Size (L)'] <= 3.5)]
num_vehicles = filtered_vehicles.shape[0]

average_co2_emission = filtered_vehicles['CO2 Emissions (g/km)'].mean()

print(f"Broj vozila s veličinom motora između 2.5 i 3.5 L: {num_vehicles}")
print(f"Prosječna emisija CO2 za ta vozila: {average_co2_emission} g/km")
print("\n")

#d)
audi_vehicles = df[df['Make'] == 'Audi']

num_audi_vehicles = audi_vehicles.shape[0]
audi_vehicles_4_cylinders = audi_vehicles[audi_vehicles['Cylinders'] == 4]
average_co2_emission_audi_4_cylinders = audi_vehicles_4_cylinders['CO2 Emissions (g/km)'].mean()

print(f"Broj mjerenja za vozila proizvođača Audi: {num_audi_vehicles}")
print(f"Prosječna emisija CO2 za Audi vozila s 4 cilindra: {average_co2_emission_audi_4_cylinders} g/km")
print("\n")

#e)
#print(df['Cylinders'].unique())
valid_cylinders = [3, 4, 5, 6, 8, 10, 12, 16]

filtered_vehicles = df[df['Cylinders'].isin(valid_cylinders)]
vehicle_count_by_cylinders = filtered_vehicles['Cylinders'].value_counts()
average_co2_by_cylinders = filtered_vehicles.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()

print("Broj vozila po broju cilindara:")
print(vehicle_count_by_cylinders)
print("\nProsječna emisija CO2 po broju cilindara:")
print(average_co2_by_cylinders)
print("\n")

#f)
diesel_vehicles = df[df['Fuel Type'] == 'D']
gasoline_vehicles = df[df['Fuel Type'] == 'X']

average_city_consumption_diesel = diesel_vehicles['Fuel Consumption City (L/100km)'].mean()
average_city_consumption_gasoline = gasoline_vehicles['Fuel Consumption City (L/100km)'].mean()
median_city_consumption_diesel = diesel_vehicles['Fuel Consumption City (L/100km)'].median()
median_city_consumption_gasoline = gasoline_vehicles['Fuel Consumption City (L/100km)'].median()

print(f"Prosječna gradska potrošnja za vozila na dizel: {average_city_consumption_diesel} L/100km")
print(f"Prosječna gradska potrošnja za vozila na regularni benzin: {average_city_consumption_gasoline} L/100km")
print(f"Medijalna gradska potrošnja za vozila na dizel: {median_city_consumption_diesel} L/100km")
print(f"Medijalna gradska potrošnja za vozila na regularni benzin: {median_city_consumption_gasoline} L/100km")
print("\n")

#g)
diesel_4_cylinders = df[(df['Cylinders'] == 4) & (df['Fuel Type'] == 'D')]
vehicle_with_highest_city_consumption = diesel_4_cylinders.loc[diesel_4_cylinders['Fuel Consumption City (L/100km)'].idxmax()]
print("Vozilo s 4 cilindra koje koristi dizelski motor ima najveću gradsku potrošnju goriva:")
print(f"Proizvođač: {vehicle_with_highest_city_consumption['Make']}, Model: {vehicle_with_highest_city_consumption['Model']}, Gradska potrošnja goriva: {vehicle_with_highest_city_consumption['Fuel Consumption City (L/100km)']} L/100km")
print("\n")

#h)
manual_vehicles = df[df['Transmission'].str.contains('M', na=False)]
num_manual_vehicles = manual_vehicles.shape[0]

print(f"Broj vozila s ručnim mjenjačem: {num_manual_vehicles}")
print("\n")

#i)
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()

print("Korelacija između numeričkih veličina:")
print(correlation_matrix)

print("\nKomentar:")
for column in correlation_matrix.columns:
    for row in correlation_matrix.index:
        correlation_value = correlation_matrix.loc[row, column]
        
        if abs(correlation_value) > 0.8:
            if correlation_value > 0:
                print(f"Proporcionalna korelacija između '{row}' i '{column}': {correlation_value}")
            elif correlation_value < 0:
                print(f"Obrnuta proporcionalnost između '{row}' i '{column}': {correlation_value}")