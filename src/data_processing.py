import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def process_fraud_data(filepath, n_components=4):
    """
    Cleans data, handles dimensionality reduction to 4 features,
    and splits into train/test sets.
    """
    # 1. Load Dataset
    df = pd.read_csv(filepath)
    
    # 2. Basic Cleaning: Remove missing values and outliers
    df = df.dropna()
    
    # Assuming 'Class' is the target column for fraud
    X = df.drop('fraud', axis=1)
    y = df['fraud'].values
    
    # 3. Scaling (Essential for both PCA and Quantum Feature Maps)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Dimensionality Reduction (Constraint: max 4 features)
    # PCA is used to compress 8 features into the 4 most important principal components
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    
    # 5. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, pca

if __name__ == "__main__":
    # Test the pipeline
    try:
        X_train, X_test, y_train, y_test, _ = process_fraud_data('data/dataset.csv')
        print(f"Data processed successfully.")
        print(f"Input shape: {X_train.shape}")
        print(f"Features reduced to: {X_train.shape[1]}")
    except Exception as e:
        print(f"Error: {e}. Ensure dataset.csv is in the data/ folder.")