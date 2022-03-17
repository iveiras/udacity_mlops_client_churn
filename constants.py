'''
File that contains all the constants needed to run churn_library
and churn_script_logging_and_tests modules

Author: Iago Veiras
Date: March 2022
'''

# churn_library constants
FIGSIZE_EDA = (20, 10)
CHURN_HIST_PTH = './images/eda/churn_histplot.png'
AGE_HIST_PTH = './images/eda/customerage_histplot.png'
MARITAL_HIST_PTH = './images/eda/maritalstatus_histplot.png'
TRANS_HIST_PTH = './images/eda/totaltransactions_distplot.png'
CMAP_HEATMAP = 'Dark2_r'
CORR_HEATMAP_PTH = './images/eda/corr_heatmap.png'
TEST_SIZE = 0.3
RANDOM_STATE = 42
FIGSIZE_CR = (6, 5)
FONTSIZE = 10
FONTPROPERTIES = 'monospace'
RF_CR_PTH = './images/results/rf_classreport.png'
LR_CR_PTH = './images/results/lr_classreport.png'
FIGSIZE_FEATURE = (15, 15)
ROTATION_FEATURE = 90
PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}
CV = 5
FIGSIZE_ROC = (15, 8)
ALPHA = 0.8
RPC_PTH = './images/results/roc_curves.png'
SHAP_PTH = './images/results/shap_values.png'
RFC_MODEL_PTH = './models/rfc_model.pkl'
LR_MODEL_PTH = './models/logistic_model.pkl'

# churn_script_logging_and_tests constants
INPUT_DATA_PTH = './data/bank_data.csv'
CATEGORY_LST = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]
KEEP_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Gender_Churn',
    'Education_Level_Churn',
    'Marital_Status_Churn',
    'Income_Category_Churn',
    'Card_Category_Churn']
FEATURE_PTH = './images/results/rfc_feature_importance.png'
