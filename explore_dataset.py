import pandas as pd

# Load dataset
data = pd.read_csv("dataset/Salary_Data.csv")

# Show first 5 rows
print("First 5 rows of dataset:")
print(data.head())

print("\n-----------------------")

# Dataset shape
print("Dataset Shape:")
print(data.shape)

print("\n-----------------------")

# Column names
print("Columns in dataset:")
print(data.columns)

print("\n-----------------------")

# Dataset information
print("Dataset Info:")
print(data.info())

print("\n-----------------------")

# Check missing values
print("Missing Values:")
print(data.isnull().sum())