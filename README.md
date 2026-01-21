# Predictive Modeling for Drug Discovery via Virtual Screening

## Project Overview

This project investigates the use of machine learning methods to predict the biological
activity of chemical compounds in the context of virtual screening for drug discovery.
The task is formulated as a binary classification problem, where each compound is labeled
as either biologically active or inactive against a given target.

Virtual screening is an important computational technique in early-stage drug discovery.
It allows researchers to prioritize promising candidate molecules before conducting
expensive and time-consuming laboratory experiments. The goal of this project is to
simulate such a workflow and evaluate how different machine learning models perform on
this task.

---

## Objectives

The main objectives of this project are:

- To develop a machine learning pipeline for virtual screening
- To compare a simple baseline model with more advanced classifiers
- To evaluate model performance using standard classification metrics
- To analyze model behavior, generalization, and robustness
- To examine feature importance and model interpretability where applicable

---

## Dataset

Source: Kaggle – Drug Discovery Virtual Screening Dataset  
Link: https://www.kaggle.com/datasets/shahriarkabir/drug-discovery-virtual-screening-dataset

Description:  
The dataset contains 2,000 chemical compounds described by molecular properties,
protein characteristics, and binding affinity measurements. It is designed to resemble
real pharmaceutical research data while preserving confidentiality.

Target Variable:

- Activity label:
  - 1 indicates an active compound
  - 0 indicates an inactive compound

---

## Modeling Approaches

### Baseline Model

Logistic Regression  
A simple and interpretable linear classifier is used as a baseline. Its performance serves
as a reference for evaluating more complex models.

### Tree-Based Models

Random Forest and Gradient Boosting classifiers are employed to capture non-linear
relationships between molecular features and biological activity. These models are widely
used in cheminformatics due to their robustness and ability to model feature interactions.

### Neural Network

A feed-forward neural network will be implemented using a deep learning framework such
as TensorFlow/Keras or PyTorch. This model explores whether deeper architectures can learn
more complex feature representations. Regularization techniques are applied to mitigate
overfitting.

All models are trained using the same train–validation–test split to ensure a fair
comparison. Hyperparameter tuning is performed using cross-validation where appropriate.

---

## Evaluation

Model performance is assessed using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Additional analyses include:

- Confusion matrices
- Learning curves and loss trajectories
- Error analysis to examine false positives and false negatives

---

## Interpretability and Analysis

For tree-based models, feature importance scores are computed to identify which molecular
descriptors contribute most strongly to the prediction of biological activity. This
analysis provides insight into the physicochemical properties associated with compound
effectiveness.

Error analysis is performed to assess whether the models struggle more with identifying
active compounds or correctly filtering inactive ones, which is a critical consideration
in virtual screening applications.

---

## Expected Outcomes

- Advanced models are expected to outperform the baseline Logistic Regression model due
  to their ability to capture non-linear relationships.
- A systematic comparison of interpretability, computational complexity, and predictive
  performance is provided.
- The final result is a reusable machine learning pipeline that simulates a real-world
  virtual screening workflow for early-stage drug discovery.

---

## Project Structure

drug-discovery-virtual-screening/
│
├── data/
│   ├── raw/                         # Original dataset
│   └── processed/                   # Cleaned and prepared data
│
├── notebooks/                       # Jupyter notebooks for analysis
│   ├── 00_setup_and_data_check.ipynb
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_baseline_logistic_regression.ipynb
│   ├── 03_tree_models_rf_gb.ipynb
│   ├── 04_neural_network.ipynb
│   └── 05_model_comparison_and_error_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_prep.py                 # Data loading and preprocessing
│   ├── train.py                     # Model training
│   ├── evaluate.py                  # Model evaluation
│   └── utils.py                     # Helper functions
│
├── results/                         # Metrics, figures, and outputs
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore file
├── LICENSE                          # MIT License
└── README.md                        # Project documentation

---

## Data

The dataset can be downloaded from Kaggle:
https://www.kaggle.com/datasets/shahriarkabir/drug-discovery-virtual-screening-dataset

To download manually:
1. Visit the Kaggle link above
2. Download the dataset file
3. Place the file in the `data/raw/` directory

---

## Installation

To install the required dependencies, run:

pip install -r requirements.txt

Python version 3.9 or higher is recommended.

---

## Quick Start

1. Clone the repository:

git clone https://github.com/yourusername/drug-discovery-virtual-screening.git
cd drug-discovery-virtual-screening

2. Download the dataset from Kaggle and place it in:

data/raw/

3. Install dependencies:

pip install -r requirements.txt

4. Run the analysis notebooks in order:

- Start with 00_setup_and_data_check.ipynb
- Continue with 01_eda_and_preprocessing.ipynb
- Train models in notebooks 02–04
- Compare models and analyze errors in 05_model_comparison_and_error_analysis.ipynb

Final outputs and figures will be saved in the `results/` directory.

---

## Key Findings

(To be populated after model training and evaluation)

- Best performing model:
- Most important features:
- Key challenges identified:
- Implications for virtual screening:

---

## Author

Milica Jeftić  
Bioinformatics student  
University of Primorska – FAMNIT  
Student ID: 89211255

---

## License

This project is released under the MIT License.
