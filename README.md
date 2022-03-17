# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project aims to to identify credit card customers that are most likely to churn, based on the dataset available in [Kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers). This project implements two different churn predictors: a Random Forest Classifier and a Logistic Regression method.

The code included in the project complete the process for solving the data science problem mentioned:

1. Exploratory Data Analysis (EDA)
2. Feature Engineering
3. Models Training
4. Prediction
5. Model Evaluation

This project also includes a script in order to test and log all the functions implemented.


## Running Files
In order to run the main script and perform the data science project, the `churn_library.py` can be executed with the following command:

```
ipython churn_library.py
```

Executing the main script generates several graphic outputs (EDA/feature importance/shap/ROC plots and classification reports) as well as the two trained models.

There is also another script that allows the user to test and log the results of each defined function from `churn_library.py`. This other script can be executed with the following command:

```
ipython churn_script_logging_and_tests.py
```

Apart from generating the same outputs than the main script, it also logs the process and stores it in `./logs/churn_library.log`.


## Dependencies

This project was developed in `Python 3.6.3`, and the following libraries are needed to correctly run it:

```
joblib >= 0.11
matplotlib >= 2.1.0
numpy >= 1.12.1
pandas >= 0.23.3
scikit-learn == 0.22
seaborn >= 0.8.1
shap >= 0.40.0
```


## Author and Date

__Iago Veiras Lens__ *(based on Udacity original code)* - March 2022
