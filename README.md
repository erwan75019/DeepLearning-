#  ECG200 Classification with Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Project](https://img.shields.io/badge/Type-DeepLearning-informational)

---

##  Project Overview

This project focuses on **binary classification of ECG signals** to detect **myocardial infarction (heart attack)** using Deep Learning models.

Three architectures are implemented and compared:

* **MLP (Multi-Layer Perceptron)**
* **CNN (1D Convolutional Neural Network)**
* **LSTM (Long Short-Term Memory)**

 Goal: evaluate model performance on **short biomedical time-series data**

---

##  Models

###  MLP (Best Performing)

* Dense layers + Dropout
* Very stable across runs
* Strong performance on small datasets

###  CNN (1D)

* 3 Conv1D layers (kernel sizes: 7, 5, 3)
* Batch Normalization + MaxPooling
* Feature extraction from signal patterns

###  LSTM

* Sequential modeling of time dependencies
* Limited by dataset size

---

##  Methodology

* Data normalization using `MinMaxScaler`
* Modular architecture:

  * `models/` → model definitions
  * `utils/` → training, evaluation, preprocessing
* Central execution via `main.py`
* Robust evaluation:

  * Multiple runs per model
  * Aggregated metrics

---

##  Results (Key Insights)

* **MLP outperforms CNN and LSTM** on this dataset
* Simpler models can outperform complex ones on small datasets
* CNN captures local patterns but lacks stability
* LSTM requires more data to fully exploit temporal dependencies

---

##  Evaluation Metrics

* Accuracy
* Precision / Recall / F1-score
* ROC Curve & AUC
* Confusion Matrix

Results are automatically generated in:

```
results/
├── graphs/
├── metrics/
└── reports/
```

---

##  Repository Structure

```
.
├── models/
├── utils/
├── results/
├── main.py
├── requirements.txt
└── README.md
```

---

##  Installation

```
git clone https://github.com/erwan75019/DeepLearning-.git
cd DeepLearning-
pip install -r requirements.txt
```

---

##  Run the project

```
python main.py
```

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* NumPy / Pandas
* Matplotlib

---

##  What This Project Demonstrates

✔ Deep learning applied to time-series data
✔ Comparison of multiple neural architectures
✔ Clean and modular ML project structure
✔ Reproducible experimentation pipeline
✔ Automated evaluation and visualization

---

##  Future Improvements

* Hyperparameter tuning (Optuna / GridSearch)
* Larger ECG datasets
* Model explainability (SHAP, Grad-CAM)
* Experiment tracking (MLflow / Weights & Biases)
* Dockerization

---

## Author

Erwan
Engineering student – Data Science & AI

---

##  Recruiter Note

This project demonstrates my ability to design, implement and evaluate deep learning models on a real-world classification problem, with a strong focus on **performance, reproducibility and clean code structure**.
