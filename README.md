# House Price Prediction

A machine learning project that predicts California housing prices using the classic California Housing dataset. Built with scikit-learn, this project walks through the full ML pipeline — from data loading and exploration to preprocessing, feature engineering, and model training.

---

## Project Overview

This project is based on the hands-on ML workflow from *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron. It covers:

- Exploratory data analysis (EDA)
- Stratified train/test splitting
- Custom feature engineering
- Full preprocessing pipeline with scikit-learn
- Comparing multiple regression models

---

## Dataset

The dataset used is the **California Housing Dataset** sourced from:
```
https://github.com/ageron/data/raw/main/housing.tgz
```
It contains **20,640 districts** across California with features like median income, house age, population, and geographic coordinates.

| Feature | Description |
|---|---|
| `longitude` / `latitude` | Geographic location |
| `housing_median_age` | Median age of houses |
| `total_rooms` / `total_bedrooms` | Rooms per district |
| `population` / `households` | People per district |
| `median_income` | Median income (in $10,000s) |
| `ocean_proximity` | Categorical location label |
| `median_house_value` | **Target variable** |

---

## Tech Stack

- Python 3.10+
- pandas, numpy, matplotlib
- scikit-learn

---

## ML Pipeline

The preprocessing pipeline handles:

- **Imputation** — fills missing values with median
- **Feature Engineering** — adds `rooms_per_house`, `bedrooms_ratio`, `people_per_house`
- **Log Transformation** — applied to heavy-tailed features
- **Cluster Similarity** — uses KMeans to encode geographic proximity
- **Scaling** — StandardScaler on all numeric features
- **One-Hot Encoding** — for `ocean_proximity`

---

## Models & Results

| Model | Training RMSE |
|---|---|
| Linear Regression | ~$68,000 |
| Decision Tree | ~$0 (overfits) |
| Random Forest | ~$17,000 ✅ |

> RMSE = Root Mean Squared Error — lower is better. Random Forest performs best on this dataset.

---

## How to Run

### Option 1 — Google Colab (Recommended)
1. Open [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Open Notebook → GitHub**
3. Paste this repo URL
4. Click **Runtime → Run All**

### Option 2 — Local Jupyter
```bash
git clone https://github.com/VanshRana1232/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
jupyter notebook house_prediction.ipynb
```

---

## 📂 Project Structure

```
house-price-prediction/
│
├── house_prediction.ipynb   # Main notebook
├── requirements.txt         # Dependencies
└── README.md
```

---

## 👤 Author

**Vansh Partap Singh**  
[GitHub](https://github.com/VanshRana1232)
[Linkedin](https://www.linkedin.com/in/vansh-partap-singh-9069b7284/)
