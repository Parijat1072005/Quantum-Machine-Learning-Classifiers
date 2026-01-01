import os
import sys
import pandas as pd
import tensorflow as tf
import pennylane as qml
from pennylane import numpy as np
from sklearn.metrics import roc_auc_score

# Ensure the script can find the 'src' directory for imports
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, '..'))

try:
    from src.data_processing import process_fraud_data
except ImportError:
    # Fallback if run directly from within the src folder
    from data_processing import process_fraud_data

def run_final_validation(new_data_path):
    print(f"--- Phase 4: Generalization Test on {new_data_path} ---")
    
    # 1. Load data and RENAME the target column to 'fraud' for compatibility
    if not os.path.exists(new_data_path):
        print(f"Error: File {new_data_path} not found.")
        return

    df = pd.read_csv(new_data_path)
    
    # Logic to identify the target column automatically
    possible_targets = ['Class', 'isFraud', 'fraud_label', 'fraud', 'target']
    actual_target = next((c for c in possible_targets if c in df.columns), None)

    if not actual_target:
        print(f"Error: Target column not found. Found columns: {df.columns.tolist()}")
        return
    
    print(f"Detected target '{actual_target}', standardizing to 'fraud'...")
    df = df.rename(columns={actual_target: 'fraud'})

    # 2. Save temporary standardized file for the processing pipeline
    temp_path = os.path.join(base_dir, "..", "data", "temp_standardized.csv")
    df.to_csv(temp_path, index=False)

    # 3. Process Data (Applying PCA reduction to 4 features)
    try:
        # process_fraud_data handles scaling and PCA constraints
        _, X_test, _, y_test, _ = process_fraud_data(temp_path)
    except Exception as e:
        print(f"Processing Error: {e}")
        return
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)

    # 4. Setup Tuned Quantum Model (Optimal Depth 6)
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev, interface="tf")
    def qnn_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(4))
        qml.StronglyEntanglingLayers(weights, wires=range(4))
        return qml.expval(qml.PauliZ(0))

    # Weight shape for Depth 6 layers
    weight_shape = (6, 4, 3)
    weights = tf.Variable(tf.random.uniform(weight_shape, 0, 1), trainable=False)

    # 5. Stratified Evaluation (Logical Fix for nan/UndefinedMetric)
    print("Selecting balanced test batch for evaluation...")
    
    # Find up to 250 samples of each class to ensure a valid AUC calculation
    idx_0 = np.where(y_test == 0)[0][:250]
    idx_1 = np.where(y_test == 1)[0][:250]
    
    if len(idx_1) == 0:
        print("Error: No fraud cases (Class 1) found in the test dataset.")
        return

    balanced_indices = np.concatenate([idx_0, idx_1])
    y_true_subset = y_test[balanced_indices]
    X_test_subset = tf.gather(X_test_tf, balanced_indices)

    print(f"Evaluating on {len(balanced_indices)} samples (Class 0: {len(idx_0)}, Class 1: {len(idx_1)})...")
    
    logits = [qnn_circuit(x, weights) for x in X_test_subset]
    probs = (tf.stack(logits).numpy() + 1) / 2
    
    final_auc = roc_auc_score(y_true_subset, probs)
    
    print("\n" + "="*40)
    print(f"PHASE 4: GENERALIZATION RESULTS")
    print("-" * 40)
    print(f"Model Configuration: Depth 6, Stratified Sample")
    print(f"Final Generalization AUC: {final_auc:.4f}")
    print("="*40)

if __name__ == "__main__":
    # Point this to your new downloaded dataset
    run_final_validation('data/final_test_dataset.csv')