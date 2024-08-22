# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xWQQds6r2yXNPmI0V6u_G0b-l55-rMyW

Jakub Barcik, 3, informatyka
"""

import pandas as pd
import numpy as np
import seaborn as sns

from google.colab import files
uploaded = files.upload()

auto4 = pd.read_csv("auto4.csv", index_col=False)
# Preview the first 5 lines of the loaded data
auto4.head()

# Count missing data
column_names = auto4.columns
counter = auto4[column_names].isnull().sum()
print (counter)

cols = auto4.columns[:30] # first 30 columns
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(auto4[cols].isnull(), cmap=sns.color_palette(colours))

for col in auto4.columns:
    pct_missing = np.mean(auto4[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

import missingno as msno
import matplotlib.pyplot as plt

msno.matrix(auto4)
plt.show()

cols_to_drop = ['exp_opinion']
auto4 = auto4.drop(cols_to_drop, axis=1)

auto4.head()

column_names = auto4.columns
counter = auto4[column_names].isnull().sum()
print (counter)

for col in auto4.columns:
    missing = auto4[col].isnull()
    num_missing = np.sum(missing)

    if num_missing > 0:
        print('created missing indicator for: {}'.format(col))
        auto4['{}_ismissing'.format(col)] = missing

ismissing_cols = [col for col in auto4.columns if 'ismissing' in col]
auto4['num_missing'] = auto4[ismissing_cols].sum(axis=1)


ind_missing = auto4[auto4['num_missing'] > 0].index


auto4 = auto4.drop(ind_missing, axis=0)
print(auto4)

# Count missing data
column_names = auto4.columns
counter = auto4[column_names].isnull().sum()
print (counter)

# If we want to remove additional columns
cols_to_drop = ['exp_opinion','exp_opinion_ismissing']
auto4 = auto4.drop(cols_to_drop, axis=1)
auto4.head()

# Strip "km" from the 'mileage' column for string values
auto4['mileage'] = auto4['mileage'].apply(lambda x: str(x).strip("km"))

# Convert duration to integer
auto4['mileage'] = auto4['mileage'].astype('int')

# Write an assert statement making sure of conversion
assert auto4['mileage'].dtype == 'int'

auto4.loc[auto4['year'] > 75, 'year'] = 75

# Find a duplicate rows
duplicated_rows= auto4[auto4.duplicated()]
auto4 = auto4.drop_duplicates()
print(auto4)

average_horsepower = auto4['horsepower'].mean()

print(f"Średnia ilość koni mechanicznych: {average_horsepower:.2f}")

sorted_auto = auto4.sort_values(by='mileage')

# Wybierz pierwszy rekord (najmniejsze zużycie paliwa na galonie)
car_with_least_fuel_consumption = sorted_auto.iloc[0]

# Wyświetl informacje o samochodzie z najmniejszym zużyciem paliwa
print("Samochód z najmniejszym zużyciem paliwa na galonie:")
print(car_with_least_fuel_consumption)

sns.scatterplot(x='horsepower', y='acceleration', data=auto4)
plt.xlabel('Moc silnika (horsepower)')
plt.ylabel('Przyśpieszenie')
plt.title('Zależność między mocą silnika a przyśpieszeniem')
plt.show()

# Współczynnik korelacji
correlation = auto4['horsepower'].corr(auto4['acceleration'])
print(f'Współczynnik korelacji między mocą silnika a przyśpieszeniem: {correlation:.2f}')

vehicle_count_by_origin = auto4['origin'].value_counts()

most_common_origin = vehicle_count_by_origin.idxmax()

print(f"Największa liczba pojazdów pochodzi z regionu: {most_common_origin}")

auto4.corr()
sns.heatmap(auto4.corr());