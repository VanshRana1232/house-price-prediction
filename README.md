# House Price Prediction (Machine Learning Project)

A machine learning pipeline that predicts California housing prices using Scikit-Learn.

This project demonstrates the complete workflow of a real-world machine learning project including:

* Data loading
* Data exploration
* Data preprocessing
* Feature engineering
* Data cleaning
* Building ML pipelines
* Training and evaluating a model

The project is implemented using **Python**, **Scikit-Learn**, **Pandas**, and **NumPy** inside a Jupyter Notebook.

---

## Project Objective

The goal of this project is to predict the **median house value** based on various housing features such as:

* Median income
* Total rooms
* Population
* Number of households
* Housing location attributes

The project also focuses on building a **robust preprocessing pipeline** that can automatically prepare raw data for machine learning models.

---

## Dataset

The dataset used in this project is the **California Housing Dataset**.

It contains information about housing blocks including:

* longitude
* latitude
* housing_median_age
* total_rooms
* total_bedrooms
* population
* households
* median_income
* ocean_proximity
* median_house_value (target variable)

Source:
https://github.com/ageron/data

---

## Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib

---

## Machine Learning Workflow

### 1. Data Loading

The dataset is downloaded and loaded into a Pandas DataFrame.

### 2. Data Exploration

Initial exploration is done using:

* `head()`
* `info()`
* correlation analysis

This helps understand feature distributions and relationships.

### 3. Train-Test Split

The dataset is split into training and testing sets using:

* Random sampling
* Stratified sampling based on income categories

This ensures the test set represents the data properly.

### 4. Feature Engineering

New useful features are created such as:

* `rooms_per_house`
* `bedrooms_ratio`
* `people_per_house`

These derived features help improve model performance.

### 5. Data Cleaning

Missing values are handled using:

`SimpleImputer(strategy="median")`

### 6. Feature Scaling

Numeric features are scaled using:

`StandardScaler()`

### 7. Machine Learning Pipeline

A preprocessing pipeline is built using:

`Pipeline` and `ColumnTransformer`

This automates the entire preprocessing step.

### 8. Model Training

A **Linear Regression model** is trained using the processed dataset.

### 9. Predictions

The trained model predicts house prices on sample data.

---

## Example Prediction

```
Predicted house values:
[205000, 180000, 340000, ...]
```

---

## How to Run the Project

1. Clone the repository

```
git clone https://github.com/VanshRana1232/house-price-prediction.git
cd house-price-prediction
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Open the notebook

```
jupyter notebook house_prediction.ipynb
```

4. Run all cells to reproduce the results.

---

## Project Structure

```
house-price-prediction
│
├── house_prediction.ipynb
├── README.md
└── requirements.txt
```

---

## Future Improvements

* Add more models (Random Forest, Gradient Boosting)
* Perform hyperparameter tuning
* Build a web interface using Streamlit
* Deploy the model

---

## Author

**Vansh Partap Singh**

AI/ML student passionate about machine learning, data science, and building intelligent systems.
