# Quantum-Machine-Learning-Classifiers

This project explores the application of Variational Quantum Circuits (VQC) and Hybrid Quantum Neural Networks (QNN) to identify fraudulent financial transactions. Fraud detection is traditionally difficult due to extreme class imbalance; in this dataset, only 0.17% of transactions are fraudulent.

By utilizing Angle Embedding to map classical data into a 4-qubit Hilbert space and implementing Dynamic Cost-Sensitive Learning, this project demonstrates how quantum architectures can be tuned to prioritize rare fraud events over the majority legitimate class. The pipeline follows a systematic 5-week development cycle, culminating in a robust model that generalizes across different financial datasets.

## Key Features:
**Dimensionality Reduction**: Classical preprocessing and PCA to reduce high-dimensional financial features into a quantum-ready 4-feature set.

**Hybrid Training Loop**: Integration of PennyLane's quantum simulators with TensorFlow’s optimization engine for backpropagation-based training.

**Cost-Sensitive Optimization**: Dynamic loss weighting to handle extreme data imbalance without the need for artificial oversampling.

**Systematic Fine-Tuning**: Automated grid search over circuit depths and learning rates to find the optimal quantum ansatz for a specific data distribution.

## Tech Stack
### Quantum Computing Frameworks
PennyLane: Used for designing the Variational Quantum Circuits (VQC) and managing quantum embeddings.

Default Qubit Simulator: A high-performance state-vector simulator used for local circuit execution.

### Machine Learning & Deep Learning
TensorFlow: Powers the hybrid training loop, managing the classical gradients and the Adam optimizer.

Scikit-Learn: Used for the classical Random Forest baseline, PCA dimensionality reduction, and ROC-AUC evaluation metrics.

### Data Processing & Analytics
Pandas: Primary tool for dataset manipulation, feature renaming, and handling CSV structures.

NumPy (PennyLane-wrapped): Essential for high-performance numerical operations and managing quantum weight tensors.

Matplotlib: Used for visualizing the quantum circuit architecture and plotting performance comparisons.



## Follow the steps sequentially to run the project:



Open a colab notebook.



set t4 GPU runtime



>>!git clone https://github.com/Parijat1072005/Quantum-Machine-Learning-Classifiers


Create a folder inside the Quantum-Machine-Learning-Classifiers named "Data".



Put the csv files in it, you can find it in the initial project release.



>>!pwd //if the output is /content


>>%cd Quantum-Machine-Learning-Classifiers


>>!pip install -r requirements.txt


>>%run src/data_processing.py


>>%run notebooks/1_classical_baselines.ipynb


>>%run src/quantum_models.py


>>%run src/train_qml


>>%run notebooks/2_final_quantum_performance


>>%run src/final_validation




