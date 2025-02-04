import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

df = pd.read_csv('C:\\Users\\prana\\Github\\Internships\\Excelerate AI 2025\\Week 2\\Old.csv')

# Data Cleaning
# Convert date columns to datetime
date_columns = ['Learner SignUp DateTime', 'Opportunity End Date', 'Entry created at', 
                'Apply Date', 'Opportunity Start Date', 'Date of Birth']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Remove duplicates
df.drop_duplicates(inplace=True)

today = pd.Timestamp('today')
df['Age'] = ((today - df['Date of Birth']).dt.total_seconds() / (365.25 * 24 * 60 * 60))
df['Age'] = df['Age'].fillna(-1).astype(int)  # Fill NaN with -1 before converting to int
df.loc[df['Age'] < 0, 'Age'] = np.nan  # Convert -1 back to NaN

# Remove any unreasonable ages (e.g., negative or over 100)
df.loc[(df['Age'] < 0) | (df['Age'] > 100), 'Age'] = np.nan

print("\nBasic Statistics:")
print(df.describe())

print("\nSignup Analysis by Gender:")
gender_stats = df.groupby('Gender').size()
print(gender_stats)

print("\nStatus Distribution:")
status_stats = df.groupby('Status Description').size()
print(status_stats)

plt.figure(figsize=(12, 6))

# 1. Bar Chart - Status Distribution
plt.subplot(1, 3, 1)
sns.countplot(data=df, y='Status Description')
plt.title('Status Distribution')
plt.tight_layout()

# 2. Bar Chart - Gender Distribution
plt.subplot(1, 3, 2)
sns.countplot(data=df, x='Gender')
plt.title('Gender Distribution')
plt.xticks(rotation=45)
plt.tight_layout()

# 3. Histogram - Age Distribution
plt.subplot(1, 3, 3)
sns.histplot(data=df, x='Age', bins=20)
plt.title('Age Distribution')
plt.tight_layout()
plt.show()

# Country Analysis
plt.figure(figsize=(12, 6))
country_counts = df['Country'].value_counts().head(10)
sns.barplot(x=country_counts.values, y=country_counts.index)
plt.title('Top 10 Countries')
plt.show()

# Time Analysis
plt.figure(figsize=(12, 6))
df['Signup Month'] = df['Learner SignUp DateTime'].dt.month
monthly_signups = df.groupby('Signup Month').size()
plt.plot(monthly_signups.index, monthly_signups.values)
plt.title('Monthly Signup Trend')
plt.xlabel('Month')
plt.ylabel('Number of Signups')
plt.show()

# Key Insights
print("\nKey Insights:")
print("1. Gender Distribution:", df['Gender'].value_counts(normalize=True).round(2))
print("2. Average Age:", round(df['Age'].mean(), 2))  # Fixed the round() method usage
print("3. Top Country:", df['Country'].mode()[0])
print("4. Most Common Status:", df['Status Description'].mode()[0])
print("5. Total Unique Participants:", len(df))

# Correlation Analysis
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Additional data quality checks
print("\nMissing Values:")
print(df.isnull().sum())

print("\nAge Statistics:")
print(df['Age'].describe())

# Save cleaned dataset
df.to_csv('C:\\Users\\prana\\Github\\Internships\\Excelerate AI 2025\\Week 2\\Dataset\\Week 2 Cleaned Data.csv', index=False)
