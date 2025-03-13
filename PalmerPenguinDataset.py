#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the datasets
penguins_lter = pd.read_csv('penguins_lter.csv')
penguins_size = pd.read_csv('penguins_size.csv')

# Display the first few rows of each dataset
print("First few rows of penguins_lter dataset:")
print(penguins_lter.head())
print("\nFirst few rows of penguins_size dataset:")
print(penguins_size.head())

# Data Cleaning
# Check for missing values in both datasets
missing_values_lter = penguins_lter.isnull().sum()
missing_values_size = penguins_size.isnull().sum()

print("\nMissing values in penguins_lter dataset:\n", missing_values_lter)
print("\nMissing values in penguins_size dataset:\n", missing_values_size)

# Fill missing values in numerical columns with the mean
penguins_lter['Culmen Length (mm)'].fillna(penguins_lter['Culmen Length (mm)'].mean(), inplace=True)
penguins_lter['Culmen Depth (mm)'].fillna(penguins_lter['Culmen Depth (mm)'].mean(), inplace=True)
penguins_lter['Flipper Length (mm)'].fillna(penguins_lter['Flipper Length (mm)'].mean(), inplace=True)
penguins_lter['Body Mass (g)'].fillna(penguins_lter['Body Mass (g)'].mean(), inplace=True)

penguins_size['culmen_length_mm'].fillna(penguins_size['culmen_length_mm'].mean(), inplace=True)
penguins_size['culmen_depth_mm'].fillna(penguins_size['culmen_depth_mm'].mean(), inplace=True)
penguins_size['flipper_length_mm'].fillna(penguins_size['flipper_length_mm'].mean(), inplace=True)
penguins_size['body_mass_g'].fillna(penguins_size['body_mass_g'].mean(), inplace=True)

# Fill missing values in categorical columns with the mode
penguins_lter['Sex'].fillna(penguins_lter['Sex'].mode()[0], inplace=True)
penguins_size['sex'].fillna(penguins_size['sex'].mode()[0], inplace=True)

# Confirm that missing values have been handled
missing_values_lter = penguins_lter.isnull().sum()
missing_values_size = penguins_size.isnull().sum()

print("\nRemaining missing values in penguins_lter dataset:\n", missing_values_lter)
print("\nRemaining missing values in penguins_size dataset:\n", missing_values_size)

# Exploratory Data Analysis (EDA)
# Statistical summary of penguins_lter dataset
summary_lter = penguins_lter.describe(include='all')
# Statistical summary of penguins_size dataset
summary_size = penguins_size.describe(include='all')

print("\nStatistical summary of penguins_lter dataset:\n", summary_lter)
print("\nStatistical summary of penguins_size dataset:\n", summary_size)

# Handle infinite values in numerical columns before plotting
penguins_lter.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
penguins_size.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

# Histograms for numerical columns
numerical_cols = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
penguins_lter[numerical_cols].hist(bins=15, figsize=(15, 6), layout=(2, 2))
plt.show()

# Box plots for numerical columns
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for idx, col in enumerate(numerical_cols):
    sns.boxplot(y=penguins_lter[col], ax=axes[idx // 2, idx % 2])
fig.suptitle('Box Plots for Numerical Columns')
plt.tight_layout()
plt.show()

# Scatter plots to understand relationships between numerical variables only
penguins_lter_cleaned = penguins_lter[numerical_cols + ['Species']].dropna()
sns.pairplot(penguins_lter_cleaned, hue='Species')
plt.show()

# Feature Engineering
penguins_lter['Culmen Length x Depth'] = penguins_lter['Culmen Length (mm)'] * penguins_lter['Culmen Depth (mm)']
penguins_lter['Body Mass Category'] = pd.cut(penguins_lter['Body Mass (g)'], bins=[2000, 3000, 4000, 6000], labels=['Light', 'Medium', 'Heavy'])

# Convert categorical columns to numerical
penguins_lter['Species'] = penguins_lter['Species'].astype('category').cat.codes
penguins_lter['Sex'] = penguins_lter['Sex'].astype('category').cat.codes
penguins_lter['Island'] = penguins_lter['Island'].astype('category').cat.codes
penguins_lter['Body Mass Category'] = penguins_lter['Body Mass Category'].astype('category').cat.codes

# Model Building
# Select features and target
features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Sex', 'Island', 'Culmen Length x Depth', 'Body Mass Category']
X = penguins_lter[features]
y = penguins_lter['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Model Evaluation
# Predict and evaluate the model
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

