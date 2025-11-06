# Credit-Card-Fraud-Detection
## Overview

This project presents a thorough analysis and implementation of a machine learning pipeline to detect fraudulent credit card transactions. Using the publicly available Kaggle dataset, the notebook guides you through exploratory data analysis (EDA), preprocessing steps, model training, evaluation, and results visualization.

## Dataset

- **Source:** [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Description:** The dataset contains transactions made by credit cards in September 2013 by European cardholders. It includes 284,807 transactions, among which 492 are frauds. Features are numerical and result from a PCA transformation (V1–V28), plus `Time`, `Amount`, and the target variable `Class` (0 for genuine, 1 for fraud).

## Repository Structure

```
├── Credit Card Fraud_Detection.ipynb  # Jupyter notebook with full analysis
├── README.md                         # Project overview and instructions
├── requirements.txt                  # Python dependencies
└── data/
    └── creditcard.csv                # Raw dataset (downloaded separately)
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```
2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download the dataset**
   - Sign in to [Kaggle](https://www.kaggle.com), navigate to the dataset, and download `creditcard.csv`.
   - Place the file in the `data/` directory.

## Usage

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
2. **Open** `Credit Card Fraud_Detection.ipynb` and run all cells.
3. **Explore the analysis**:
   - Data loading and inspection
   - Exploratory Data Analysis (visualizations, class imbalance check)
   - Feature scaling and preprocessing
   - Model training (Logistic Regression, Random Forest, XGBoost, etc.)
   - Model evaluation (confusion matrix, precision, recall, ROC-AUC)

## Key Steps and Methodology

1. **Exploratory Data Analysis (EDA)**
   - Examine class distribution of transactions
   - Visualize correlations and outliers
2. **Preprocessing**
   - Scale `Amount` and `Time` features using StandardScaler
   - Split into training and test sets
   - Address class imbalance using techniques such as SMOTE or class weighting
3. **Modeling**
   - Train multiple classifiers (e.g., Logistic Regression, Random Forest, XGBoost)
   - Perform hyperparameter tuning with GridSearchCV
4. **Evaluation**
   - Use precision, recall, F1-score, and ROC-AUC to assess model performance
   - Compare models to select the best performer

## Results

- Best model: **XGBoost** with ROC-AUC of _0.99_ on the test set.
- Precision and recall curves demonstrate strong performance in identifying fraudulent transactions while minimizing false positives.

## Future Work

- Experiment with advanced deep learning architectures (e.g., autoencoders, LSTM-based models).
- Deploy the model as a REST API using Flask or FastAPI.
- Integrate real-time streaming data processing for live fraud detection.

## Requirements

- Python >= 3.7
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- matplotlib
- seaborn

*Install all requirements via*:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Authors

- **Bradley Agwa**
- **Candence Chumba**
- **Tyrone Darren**
- **Melissa Wachira**
- **Mishiel Nasambu Wakoli**
- **Joseph Kamau**  


---


