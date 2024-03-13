# Churn Exploratory Analysis Project

## Overview


This project aims to predict customer churn using machine learning techniques. The dataset contains information about bank customers and whether they have exited (churned) or not. 

The goal is to build a model that accurately identifies customers at risk of churning, allowing the bank to take proactive measures to retain them. 

It also aims to explore and analyze churn data to understand the distribution between variables, between classes, and within the classes, including correlation and pair plot analysis, as well as feature importance assessment using logistic regression and random forest models. 

### Dataset Information
Observations: 10,000

Features:
'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Geography', 'Gender', 'HasCrCard', 'IsActiveMember'

Target: 'Exited' (churned)

## Project Structure
The project consists of a single Python file containing all functions:

churn_analysis.py: Python script containing functions for data exploration, correlation analysis, feature importance assessment, and undersampling techniques.

## Code Structure
The project is organized into three main functions within a Python file:

ETL Function: Extracts, transforms, and loads the dataset.

Exploratory and Initial Machine Learning Function: Performs exploratory data analysis and builds initial machine learning models.

Balancing Dataset Function: Implements two approaches to address dataset imbalance:

Manual solution using oversampling and undersampling.

Undersampling alone.

## Setup and Dependencies
To run the functions in churn_analysis.py, you'll need the following dependencies:

Python 3.x
NumPy
pandas
matplotlib
seaborn
scikit-learn


## Results
The functions provided in churn_analysis.py enable users to perform various analyses, including data exploration, correlation analysis, feature importance assessment, and undersampling techniques. By leveraging these functions, users can gain insights into churn data and develop predictive models for churn prediction.

## Insights

Imbalanced Dataset: The dataset is imbalanced, with only 20% of observations representing churned customers.

Geographical Analysis: While most customers are from France, Germany has the highest churn rate.

Gender Disparity: Although the dataset is almost balanced in terms of gender, female clients are more likely to exit, based on the data available.

Age Distribution: The age distribution is almost normal, with churned customers having a higher average age than non-churned customers.

![Correlation Heatmap](https://github.com/gustavomfsantos/Churn_Project/blob/main/Plots/Interaction/Correlation.png)

![Correlation Heatmap](https://github.com/gustavomfsantos/Churn_Project/blob/main/Plots/Features/Features_Importance_RF.png)

## Model Performance

Manual Solution (Ensemble Approach):

Lower general accuracy but excellent recall score.

Undersampling Alone:

Slightly better overall accuracy but lower recall.


## Usage

To run the project:

Clone the repository.
Ensure you have Python installed.
Run the Python file containing the project functions.

## Future Work
Enhance the functionality of existing functions and add new functions to support additional analyses.
Optimize the performance of machine learning models for churn prediction.
Develop a user-friendly interface or dashboard for easy access to analysis tools and results.
Complement this ReadME files with more robust content and images
