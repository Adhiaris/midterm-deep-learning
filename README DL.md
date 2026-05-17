# Deep Learning Midterm Projects
Name: Adhiaris M Azka
NIM: 1103220143
Class: TK-46-G13

---

## Repository Structure

```
├── Midterm-Clustering.ipynb           # Clustering with deep learning-based representations
├── midterm-regresin-updated.ipynb     # Regression with Neural Network (Keras + MLflow)
└── midterm-fraud-detection.ipynb      # Fraud detection using Neural Network classifier
```

---

## Project Overview

### 1. Clustering — `Midterm-Clustering.ipynb`
Unsupervised learning notebook that segments data into meaningful groups using deep learning-based approaches.

- **Framework:** TensorFlow / Keras
- **Libraries:** `scikit-learn`, `optuna`, `matplotlib`

---

### 2. Regression — `midterm-regresin-updated.ipynb`
Supervised regression task predicting a continuous target variable from 89 features using a fully connected neural network.

- **Model:** Neural Network (Keras Sequential)
- **Architecture:**
  - Input layer (89 features)
  - Dense(256, activation=relu) + Dropout(0.3)
  - Dense(128, activation=relu) + Dropout(0.3)
  - Dense(1) — output layer
- **Optimizer:** Adam
- **Loss:** Mean Squared Error (MSE)
- **Training:** 20 epochs, batch size 512
- **Metrics:**

| Metric | Value     |
|--------|-----------|
| MSE    | 8705.54   |
| RMSE   | 93.30     |
| MAE    | 73.96     |
| R²     | -78.83    |

- **Experiment Tracking:** MLflow (`midterm-regression` experiment, run: `NeuralNetwork`)
- **Preprocessing:** Median imputation, IQR-based outlier removal, StandardScaler
- **Split:** 80/20 train-test

---

### 3. Fraud Detection — `midterm-fraud-detection.ipynb`
Binary classification task detecting fraudulent financial transactions using a deep neural network.

- **Model:** Neural Network (Keras Sequential)
- **Framework:** TensorFlow / Keras
- **Dataset:** IEEE-CIS Fraud Detection (50% sample, stratified split)
- **Imbalance Handling:** SMOTE (oversampling)
- **Experiment Tracking:** MLflow (`midterm-fraud-detection` experiment)
- **Preprocessing:** LabelEncoding, Median imputation, float32/int32 type casting

---

## Tech Stack

| Tool               | Purpose                             |
|--------------------|-------------------------------------|
| `tensorflow`       | Deep learning framework             |
| `keras`            | Neural network model building       |
| `imbalanced-learn` | SMOTE for class imbalance           |
| `mlflow`           | Experiment tracking & model logging |
| `pandas`, `numpy`  | Data manipulation                   |
| `matplotlib`       | Visualization                       |

---

## Datasets

| Notebook        | Dataset                                               |
|-----------------|-------------------------------------------------------|
| Clustering      | Custom unlabeled dataset                              |
| Regression      | `midterm-regresi-dataset.csv` (87,557 rows x 90 cols) |
| Fraud Detection | `train_transaction.csv` (IEEE-CIS Fraud Detection)    |

---

## MLflow Experiments

| Experiment Name           | Model          | Run Name       |
|---------------------------|----------------|----------------|
| `midterm-regression`      | Neural Network | NeuralNetwork  |
| `midterm-fraud-detection` | Neural Network | NeuralNetwork  |

---

## How to Navigate

1. Clone the repository and open the notebooks in [Google Colab](https://colab.research.google.com/) or JupyterLab.
2. Each notebook is self-contained — install dependencies by running the `!pip install` cell at the top.
3. Notebooks follow this general structure:
   - Library installation and imports
   - Dataset loading and preprocessing
   - **Deep Learning Model** section (focus of this repo)
   - Evaluation metrics and visualizations
   - MLflow experiment logging
