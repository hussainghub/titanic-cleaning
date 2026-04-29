import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
print(f"dataset: {df.shape} rows, {df.shape[1]} cols")
print("\nmissing values")
print(df.isnull().sum())
df['Age'] = df['Age'].fillna(df['Age'].median())
df.dropna(subset=['Embarked'],inplace=True)

print("cleaned survival by class: ")
print(df.groupby('Pclass')['Survived'].mean().round(3))

df.to_csv('titanic.csv', index = False)
print ("saved cleaned data")
