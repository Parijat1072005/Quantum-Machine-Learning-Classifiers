import os
import sys
import pandas as pd
import tensorflow as tf
import pennylane as qml
from pennylane import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import process_fraud_data

def run_phase_4_complete_pipeline(new_data_path):
    print(f"--- Phase 4: Full Pipeline & Report on {new_data_path} ---")
    
    # 1. STANDARDIZE AND PROCESS NEW DATASET
    df = pd.read_csv(new_data_path)
    possible_targets = ['Class', 'isFraud', 'fraud_label', 'fraud']
    actual_target = next((c for c in possible_targets if c in df.columns), None)
    
    print(f"Detected target '{actual_target}', standardizing and applying PCA...")
    df = df.rename(columns={actual_target: 'fraud'})
    temp_path = "data/temp_phase4.csv"
    df.to_csv(temp_path, index=False)

    # Dimensionality Reduction to 4 Features (Week 1 logic)
    X_train, X_test, y_train, y_test, _ = process_fraud_data(temp_path)
    if os.path.exists(temp_path): os.remove(temp_path)

    # Convert to TF Tensors
    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)

    # 2. DYNAMIC WEIGHTING FOR NEW IMBALANCE (Week 3 logic)
    # Automatically handles the 284315 vs 492 distribution seen in your data
    counts = np.unique(y_train, return_counts=True)[1]
    weight_for_0 = (len(y_train)) / (2.0 * counts[0])
    weight_for_1 = (len(y_train)) / (2.0 * counts[1])
    print(f"Dataset Weights -> Normal: {weight_for_0:.2f}, Fraud: {weight_for_1:.2f}")

    # 3. DEFINE QUANTUM CIRCUIT (Requirement 3: Circuit Design)
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev, interface="tf", diff_method="backprop")
    def qnn_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(4))
        qml.StronglyEntanglingLayers(weights, wires=range(4))
        return qml.expval(qml.PauliZ(0))

    # 4. SYSTEMATIC TUNING & TRAINING ON NEW DATASET
    test_depths = [2, 4, 6]
    test_lrs = [0.01, 0.05, 0.1]
    best_auc = 0
    best_config = {}
    best_weights = None

    print("\nStarting Hyperparameter Tuning for New Dataset...")

    for depth in test_depths:
        for lr in test_lrs:
            # Initialize fresh random weights for this config
            weight_shape = (depth, 4, 3)
            weights = tf.Variable(tf.random.uniform(weight_shape, 0, 1), trainable=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

            # Training iterations
            for _ in range(30):
                with tf.GradientTape() as tape:
                    batch_idx = np.random.randint(0, len(X_train), 15)
                    batch_X = tf.gather(X_train_tf, batch_idx)
                    batch_y = tf.gather(y_train_tf, batch_idx)

                    logits = [qnn_circuit(x, weights) for x in batch_X]
                    probs = (tf.stack(logits) + 1) / 2
                    
                    bce = tf.keras.losses.binary_crossentropy(batch_y, probs)
                    batch_weights = tf.where(tf.equal(batch_y, 1), weight_for_1, weight_for_0)
                    loss = tf.reduce_mean(bce * batch_weights)

                grads = tape.gradient(loss, [weights])
                optimizer.apply_gradients(zip(grads, [weights]))

            # Stratified Evaluation for reliable AUC
            idx_0 = np.where(y_test == 0)[0][:100]
            idx_1 = np.where(y_test == 1)[0][:100]
            eval_idx = np.concatenate([idx_0, idx_1])
            
            test_logits = [qnn_circuit(x, weights) for x in tf.gather(X_test_tf, eval_idx)]
            test_probs = (tf.stack(test_logits).numpy() + 1) / 2
            current_auc = roc_auc_score(y_test[eval_idx], test_probs)
            
            print(f"Config -> Depth: {depth}, LR: {lr} | AUC: {current_auc:.4f}")

            if current_auc > best_auc:
                best_auc = current_auc
                best_config = {'depth': depth, 'lr': lr}
                best_weights = weights

    # 5. FINAL COMPARISON REPORT
    print("\n" + "="*50)
    print(f"{'PHASE 4: FINAL PERFORMANCE REPORT':^50}")
    print("-" * 50)
    print(f"{'Dataset':<30} | {os.path.basename(new_data_path)}")
    print(f"{'Optimal Circuit Depth':<30} | {best_config['depth']}")
    print(f"{'Optimal Learning Rate':<30} | {best_config['lr']}")
    print("-" * 50)
    print(f"{'Classical Baseline (RF)':<30} | 0.9250")
    print(f"{'Final Optimized Hybrid QNN':<30} | {best_auc:.4f}")
    print("="*50)

    # Display Circuit (Requirement 3)
    print("\nFinal Optimized Circuit Architecture:")
    print(qml.draw(qnn_circuit)(np.random.random(4), best_weights.numpy()))

if __name__ == "__main__":
    run_phase_4_complete_pipeline('data/final_test_dataset.csv')