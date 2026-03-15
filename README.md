#  California House Price Prediction

A machine learning regression model that predicts median house prices across California districts using the classic California Housing dataset.

---

##  Project Overview

This project follows an end-to-end machine learning pipeline — from raw data to a fully trained and evaluated model — based on the California Housing dataset originally published in the 1990 US Census.

The goal is to predict the **median house value** of a California district given features like location, income, population, and housing characteristics.

---

##  Dataset

- **Source**: California Housing Dataset (via [Aurélien Géron's data repository](https://github.com/ageron/data))
- **Size**: ~20,000 districts
- **Target**: `median_house_value`
- **Features**: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity`

---

##  Pipeline

### 1. Data Preparation
- Loaded and explored raw housing data
- Performed **stratified train/test split** based on income categories
- Created custom features:
  - `rooms_per_house`
  - `bedrooms_ratio`
  - `people_per_house`

### 2. Preprocessing
- **Numerical features**: Imputation + Standard Scaling + Log Transformation
- **Geographical features**: `ClusterSimilarity` transformer using KMeans clustering
- **Categorical features**: One-Hot Encoding (`ocean_proximity`)
- All steps wrapped in a `ColumnTransformer` pipeline

### 3. Models Trained
| Model | RMSE (Validation) |
|---|---|
| Linear Regression | ~$68,628 |
| Decision Tree | ~$66,573 (overfitting) |
| **Random Forest** | **~$43,000**  |

### 4. Hyperparameter Tuning
- Used **RandomizedSearchCV** with `cv=3`
- Tuned:
  - `n_clusters` (3–50)
  - `max_features` (2–20)

### 5. Evaluation
- Final model evaluated on test set using **RMSE**
- **95% Confidence Interval** computed using `scipy.stats.bootstrap`
- Feature importances analyzed — `median_income` was the most important feature

---

##  Installation
```bash
git clone https://github.com/VanshRana1232/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

---

##  Usage

1. Download `final_model.pkl` from [Google Drive](https://drive.google.com/file/d/1bTe_mZOMhlvo2Ayq5fAFptYB-wILwK2y/view?usp=sharing) 
2. Open `housing_price_prediction.ipynb` in Google Colab or Jupyter
3. Load and predict:
```python
import joblib
import pandas as pd

# Load model
final_model = joblib.load("final_model.pkl")

# Predict on new data
new_data = pd.DataFrame({
    "longitude": [-122.23],
    "latitude": [37.88],
    "housing_median_age": [41],
    "total_rooms": [880],
    "total_bedrooms": [129],
    "population": [322],
    "households": [126],
    "median_income": [8.3],
    "ocean_proximity": ["NEAR BAY"]
})

prediction = final_model.predict(new_data)
print(f"Predicted House Price: ${prediction[0]:,.0f}")
```

---

##  Tech Stack

- **Python 3.10**
- **scikit-learn 1.6.1**
- **pandas 2.2.2**
- **numpy 2.0.2**
- **matplotlib 3.10.0**
- **scipy**
- **joblib**

---

##  Repository Structure
```
house-price-prediction/
│
├── housing_price_prediction.ipynb  # Main notebook
├── requirements.txt                # Dependencies
├── .gitignore                      # Ignores pkl files
└── README.md                       # This file
```

---

##  Reference

Based on Chapter 2 of [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurélien Géron.

---

##  Author

**Vansh Partap Singh** — [GitHub](https://github.com/VanshRana1232)
