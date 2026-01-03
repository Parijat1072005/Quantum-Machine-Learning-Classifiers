# Quantum-Machine-Learning-Classifiers



Follow the steps sequentially to run the project:



Open a colab notebook.



set t4 GPU runtime



>>!git clone https://github.com/Parijat1072005/Quantum-Machine-Learning-Classifiers


Create a folder inside the Quantum-Machine-Learning-Classifiers named "Data".



Put the csv files in it.



>>!pwd //if the output is /content


>>%cd Quantum-Machine-Learning-Classifiers


>>%run src/data_processing.py


>>%run notebooks/1_classical_baselines.ipynb


>>%run src/quantum_models.py


>>%run src/train_qml


>>%run notebooks/2_final_quantum_performance


>>%run src/final_validation




