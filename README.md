# House Price Prediction (Machine Learning Project)

This project builds a machine learning pipeline to predict house prices using housing data.  
It demonstrates the complete workflow of a real-world ML project including:

- Data loading
- Data exploration
- Data preprocessing
- Feature engineering
- Data cleaning
- Building ML pipelines
- Training and evaluating a model

The project is implemented using **Python**, **Scikit-Learn**, **Pandas**, and **NumPy** inside a Jupyter Notebook.

---

## Project Objective

The goal of this project is to predict the **median house value** based on various housing features such as:

- Median income
- Total rooms
- Population
- Number of households
- Housing location attributes

The project also focuses on building a **robust preprocessing pipeline** that can prepare raw data automatically for machine learning models.

---

## Dataset

The dataset used in this project is the **California Housing Dataset**.

It contains information about housing blocks including:

- longitude
- latitude
- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- median_income
- ocean_proximity
- median_house_value (target variable)

Source:  
https://github.com/ageron/data

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib

---

## 
Machine Learning Workflow

### 1. Data Loading
The dataset is downloaded and loaded into a Pandas DataFrame.

### 2. Data Exploration
Initial exploration is done using:

- `head()`
- `info()`
- correlation analysis

This helps understand feature distributions and relationships.

### 3. Train-Test Split
The dataset is split into training and testing sets using:

- Random sampling
- Stratified sampling based on income categories

This ensures the test set represents the data properly.

### 4. Feature Engineering
New useful features are created such as:

- rooms_per_house
- bedrooms_ratio
- people_per_house

These derived features improve model performance.

### 5. Data Cleaning
Missing values are handled using:
