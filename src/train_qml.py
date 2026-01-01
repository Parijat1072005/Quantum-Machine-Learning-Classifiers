import pennylane as qml
from pennylane import numpy as np
from quantum_models import q_classifier_circuit, initialize_weights
from data_processing import process_fraud_data
from sklearn.metrics import roc_auc_score

def model_predict(weights, x):
    """Converts quantum output [-1, 1] to a probability [0, 1]."""
    return (q_classifier_circuit(weights, x) + 1) / 2

def square_loss(labels, predictions):
    """Standard mean squared error loss."""
    loss = 0
    for l, p in zip(labels, predictions):
        loss += (l - p) ** 2
    return loss / len(labels)

def cost(weights, features, labels):
    """The objective function to minimize."""
    preds = [model_predict(weights, x) for x in features]
    return square_loss(labels, preds)

def train_qml_model(epochs=20, learning_rate=0.1):
    # 1. Load data
    X_train, X_test, y_train, y_test, _ = process_fraud_data('data/dataset.csv')
    
    # Use a smaller subset for training if hardware/simulation is slow
    # QML training is computationally expensive
    X_train_small = X_train[:50]
    y_train_small = y_train[:50]

    # 2. Initialize weights
    weights = initialize_weights(n_layers=3)
    
    # 3. Setup Optimizer (Classical component)
    opt = qml.AdamOptimizer(stepsize=learning_rate)
    
    print("Starting Training...")
    for epoch in range(epochs):
        # Update weights
        weights, train_cost = opt.step_and_cost(lambda w: cost(w, X_train_small, y_train_small), weights)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1} | Cost: {train_cost:0.4f}")
            
    # 4. Final Evaluation
    test_preds = [model_predict(weights, x) for x in X_test]
    auc = roc_auc_score(y_test, test_preds)
    print(f"\nTraining Complete. Final Test AUC-ROC: {auc:0.4f}")
    
    return weights

if __name__ == "__main__":
    trained_weights = train_qml_model()