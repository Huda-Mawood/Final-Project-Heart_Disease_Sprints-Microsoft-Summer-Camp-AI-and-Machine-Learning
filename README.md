# Heart_Disease_Project

> A complete project for analyzing and predicting heart disease using machine learning algorithms

---

##  Overview

This project focuses on analyzing the **Heart Disease (UCI)** dataset, performing preprocessing, feature selection, dimensionality reduction using **PCA**, training classification models, and comparing their performance. Additionally, clustering algorithms were applied to discover hidden patterns. All core steps of the pipeline were implemented **except the bonus parts (Streamlit & Ngrok)**.

---

##  Completed Work

* **Data Preprocessing & Cleaning**

  * Loaded `heart_disease.csv` into a DataFrame.
  * Handled missing values using appropriate imputation methods.
  * Encoded categorical variables (One-Hot Encoding / Label Encoding where applicable).
  * Standardized/scaled numerical features using `StandardScaler` or `MinMaxScaler`.
  * Saved the cleaned dataset for further use.

* **Exploratory Data Analysis (EDA)**

  * Plotted histograms, boxplots, and correlation heatmaps.
  * Identified distribution patterns and outliers.

* **Dimensionality Reduction — PCA**

  * Applied PCA and determined the optimal number of components using explained variance ratio.
  * Saved PCA-transformed dataset.
  * Visualized PCA scatter plot and cumulative explained variance.

* **Feature Selection**

  * Calculated feature importance using Random Forest.
  * Applied Recursive Feature Elimination (RFE) to select best predictors.
  * Used Chi-Square tests for categorical features.
  * Built a reduced dataset with selected features.

* **Supervised Learning — Classification Models**

  * Split dataset into training (80%) and testing (20%).
  * Trained models:

    * Logistic Regression
    * Decision Tree
    * Random Forest
    * Support Vector Machine (SVM)
  * Evaluated models with: Accuracy, Precision, Recall, F1-score, ROC Curve & AUC.
  * Saved evaluation results in `results/evaluation_metrics.txt`.

* **Unsupervised Learning — Clustering**

  * Applied K-Means with elbow method to determine optimal K.
  * Performed Hierarchical Clustering with dendrogram visualization.
  * Compared clusters with true labels.

* **Hyperparameter Tuning**

  * Used `GridSearchCV` and `RandomizedSearchCV` to optimize hyperparameters.
  * Compared optimized models with baseline results.

* **Model Export**

  * Saved final model and preprocessing pipeline using `joblib` (`models/final_model.pkl`).

* **Project Organization & Upload**

  * Structured files under folders (data/, notebooks/, models/, results/...).
  * Added `requirements.txt` and `.gitignore`.

## Project File Structure

```
Heart_Disease_Project/
│── data/
│   ├── heart_disease.csv
│   ├── heart_disease_cleaned.csv
│── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│── models/
│   ├── final_model.pkl
│   ├── preprocessing_pipeline.pkl
│── results/
│   ├── evaluation_metrics.txt
│   ├── roc_curves.png
│   ├── pca_variance.png
│── README.md    <- this file
│── requirements.txt
│── .gitignore
```

---

## How to Run the Project Locally

1. **Create virtual environment**:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate     # Windows
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run Jupyter notebooks**:

* Open `notebooks/` and run each notebook from `01_...` to `06_...` in order.

4. **Load saved model** (example in Python):

```python
import joblib
model = joblib.load('models/final_model.pkl')
preproc = joblib.load('models/preprocessing_pipeline.pkl')
# Use preproc.transform(...) before model.predict(...)
```

---

## Results

* Detailed metrics are available in `results/evaluation_metrics.txt`.
* Visualizations (EDA, PCA, ROC) are saved under `results/`.

## References

* UCI Heart Disease Dataset
* scikit-learn documentation
# Final-Project-Heart_Disease_Sprints-Microsoft-Summer-Camp-AI-and-Machine-Learning
