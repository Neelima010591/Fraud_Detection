import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    file_path = "data/credit_card_fraud_10k.csv"
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Drop irrelevant columns
    df = df.drop(columns=['transaction_id'])

    # (Optional) Check class imbalance
    # sns.countplot(x='is_fraud', data=df)
    # plt.show()

    # Separate columns
    categorical_cols = ['merchant_category']
    numerical_cols = df.drop(columns=['is_fraud', 'merchant_category']).columns

    # Encode categorical
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Scale numerical
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Split features & target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler, X.columns

