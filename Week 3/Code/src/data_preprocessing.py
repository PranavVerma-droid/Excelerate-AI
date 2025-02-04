import pandas as pd # type: ignore
import numpy as np  # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler  # type: ignore
from sklearn.model_selection import train_test_split            # type: ignore
from sklearn.impute import SimpleImputer                        # type: ignore
from imblearn.over_sampling import SMOTE                        # type: ignore
from datetime import datetime

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    date_columns = ['Learner SignUp DateTime', 'Opportunity End Date', 'Entry created at', 
                   'Apply Date', 'Opportunity Start Date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    df['Days_Until_Start'] = (df['Opportunity Start Date'] - df['Learner SignUp DateTime']).dt.days
    df['Application_Processing_Time'] = (df['Apply Date'] - df['Learner SignUp DateTime']).dt.days
    
    df['Date of Birth'] = pd.to_datetime(df['Date of Birth'])
    today = pd.Timestamp.now()
    df['Age'] = (today.year - df['Date of Birth'].dt.year - 
                ((today.month < df['Date of Birth'].dt.month) | 
                ((today.month == df['Date of Birth'].dt.month) & 
                (today.day < df['Date of Birth'].dt.day))))
    
    numeric_imputer = SimpleImputer(strategy='mean')
    df[['Days_Until_Start', 'Application_Processing_Time', 'Age']] = numeric_imputer.fit_transform(
        df[['Days_Until_Start', 'Application_Processing_Time', 'Age']]
    )

    le = LabelEncoder()
    categorical_columns = ['Gender', 'Country', 'Current/Intended Major', 'Status Description']

    for col in categorical_columns:
        df[col] = df[col].fillna('Unknown')
        df[col] = le.fit_transform(df[col].astype(str))
    
    drop_conditions = [
        'Withdraw', 
        'Waitlisted',
        'Rewards Award',  # Including other potential drop indicators
        1040,  # Status code for Waitlisted
        1110   # Status code for Withdrawn
    ]
    
    df['Dropped'] = (
        (df['Status Description'].isin(drop_conditions)) | 
        (df['Status Code'].isin(drop_conditions))
    ).astype(int)
    
    print(f"Class distribution:\n{df['Dropped'].value_counts()}")
    
    return df

def prepare_features(df):
    features = ['Age', 'Gender', 'Country', 'Current/Intended Major', 
               'Days_Until_Start', 'Application_Processing_Time']
    
    X = df[features]
    y = df['Dropped']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"Training set class distribution after SMOTE:\n{pd.Series(y_train_balanced).value_counts()}")
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test