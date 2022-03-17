'''
Module that includes functions needed to predict churn over Kaggle's "Credit Card customers" dataset

Author: Iago Veiras
Date: March 2022
'''

# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import constants as ct
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            input_df: pandas dataframe
    '''
    input_df = pd.read_csv(pth)
    return input_df


def calc_churn(input_df):
    '''
    calculates churn variable for the dataframe

    input:
            input_df: pandas dataframe
    output:
            input_df: pandas dataframe with churn variable

    '''
    input_df['Churn'] = input_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return input_df


def perform_eda(input_df):
    '''
    perform eda on input_df and save figures to images folder

    input:
            input_df: pandas dataframe

    output:
            None
    '''
    # churn histogram
    fig_churn = plt.figure(figsize=ct.FIGSIZE_EDA)
    ax_churn = input_df['Churn'].hist()
    ax_churn.set(
        xlabel="Client Churn",
        ylabel="No. Clients",
        title="Client Churn Histogram")
    fig_churn.savefig(ct.CHURN_HIST_PTH)

    # customer_age histogram
    fig_age = plt.figure(figsize=ct.FIGSIZE_EDA)
    ax_age = input_df['Customer_Age'].hist()
    ax_age.set(
        xlabel="Customer Age",
        ylabel="No. Clients",
        title="Customer Age Histogram")
    fig_age.savefig(ct.AGE_HIST_PTH)

    # marital_status histogram
    fig_marital = plt.figure(figsize=ct.FIGSIZE_EDA)
    ax_marital = input_df['Marital_Status'].value_counts(
        'normalize').plot(kind='bar')
    ax_marital.set(
        xlabel="Marital Status",
        ylabel="No. Clients",
        title="Client Marital Status Histogram")
    fig_marital.savefig(ct.MARITAL_HIST_PTH)

    # total_transactions distribution
    fig_totaltrans = plt.figure(figsize=ct.FIGSIZE_EDA)
    ax_trans = sns.distplot(input_df['Total_Trans_Ct'])
    ax_trans.set(xlabel="Total Transactions", ylabel="Clients %",
                 title="Client Total Transaction Distribution Plot")
    fig_totaltrans.savefig(ct.TRANS_HIST_PTH)

    # correlations heatmap
    fig_heatmap = plt.figure(figsize=ct.FIGSIZE_EDA)
    sns.heatmap(
        input_df.corr(),
        annot=False,
        cmap=ct.CMAP_HEATMAP,
        linewidths=2)
    plt.xticks(rotation=30)
    fig_heatmap.savefig(ct.CORR_HEATMAP_PTH)


def encoder_helper(input_df, category_lst, response=None):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            input_df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: list of strings of response name [optional argument that could be used for
                      naming variables or index y column]

    output:
            input_df: pandas dataframe with new columns for
    '''
    if response is None:
        response = [category + '_Churn' for category in category_lst]

    for cat, new_cat in zip(category_lst, response):
        input_df[new_cat] = input_df['Churn'].groupby(
            input_df[cat]).transform('mean')

    return input_df


def perform_feature_engineering(input_df, keep_columns):
    '''
    function that filter unused columns and splits the dataset into train and test
    subsets

    input:
              input_df: pandas dataframe
              keep_columns: list of strings of the columns to be kept

    output:
              features_train: X training data
              features_test: X testing data
              samples_train: y training data
              samples_test: y testing data
    '''
    samples = input_df['Churn']
    features = pd.DataFrame()
    features[keep_columns] = input_df[keep_columns]
    features_train, features_test, samples_train, samples_test = train_test_split(
        features, samples, test_size=ct.TEST_SIZE, random_state=ct.RANDOM_STATE)

    return features_train, features_test, samples_train, samples_test


def rf_classification_report_image(
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf):
    '''
    produces random forest classification report for training and testing results and stores
    report as image in images folder

    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_rf: training predictions from random forest
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    fig_rf = plt.figure(figsize=ct.FIGSIZE_CR)
    fig_rf.text(0.01, 0.95, str('Random Forest Train'), {
                'fontsize': ct.FONTSIZE}, fontproperties=ct.FONTPROPERTIES)
    fig_rf.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
                'fontsize': ct.FONTSIZE}, fontproperties=ct.FONTPROPERTIES)
    fig_rf.text(0.01, 0.4, str('Random Forest Test'), {
                'fontsize': ct.FONTSIZE}, fontproperties=ct.FONTPROPERTIES)
    fig_rf.text(0.01, 0.6, str(classification_report(y_train, y_train_preds_rf)), {
                'fontsize': ct.FONTSIZE}, fontproperties=ct.FONTPROPERTIES)
    fig_rf.savefig(ct.RF_CR_PTH)


def lr_classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr):
    '''
    produces logistic regression classification report for training and testing results and stores
    report as image in images folder

    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_test_preds_lr: test predictions from logistic regression

    output:
             None
    '''
    fig_log = plt.figure(figsize=ct.FIGSIZE_CR)
    fig_log.text(0.01, 0.95, str('Logistic Regression Train'), {
                 'fontsize': ct.FONTSIZE}, fontproperties=ct.FONTPROPERTIES)
    fig_log.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
                 'fontsize': ct.FONTSIZE}, fontproperties=ct.FONTPROPERTIES)
    fig_log.text(0.01, 0.4, str('Logistic Regression Test'), {
                 'fontsize': ct.FONTSIZE}, fontproperties=ct.FONTPROPERTIES)
    fig_log.text(0.01, 0.6, str(classification_report(y_test, y_test_preds_lr)), {
                 'fontsize': ct.FONTSIZE}, fontproperties=ct.FONTPROPERTIES)
    fig_log.savefig(ct.LR_CR_PTH)


def feature_importance_plot(model, features, output_pth):
    '''
    creates and stores the feature importances in pth

    input:
            model: model object containing feature_importances_
            features: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances and rearrange by importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [features.columns[i] for i in indices]

    # plot features importance
    fig_fi = plt.figure(figsize=ct.FIGSIZE_FEATURE)
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(features.shape[1]), importances[indices])
    plt.xticks(range(features.shape[1]), names, rotation=ct.ROTATION_FEATURE)
    fig_fi.savefig(output_pth)


def train_models(features_train, features_test, samples_train, samples_test):
    '''
    train, store model results: images + scores, and store models

    input:
              features_train: X training data
              features_test: X testing data
              samples_train: y training data
              samples_test: y testing data
    output:
              None
    '''
    # train models
    rfc = RandomForestClassifier(random_state=ct.RANDOM_STATE)
    lrc = LogisticRegression()

    param_grid = ct.PARAM_GRID

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=ct.CV)
    cv_rfc.fit(features_train, samples_train)

    lrc.fit(features_train, samples_train)

    # plot roc curves
    fig_roc = plt.figure(figsize=ct.FIGSIZE_ROC)
    fig_ax = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        features_test,
        samples_test,
        ax=fig_ax,
        alpha=ct.ALPHA)
    plot_roc_curve(lrc, features_test, samples_test, ax=fig_ax, alpha=ct.ALPHA)
    fig_roc.savefig(ct.RPC_PTH)

    # plot shap values
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(features_test)
    shap.summary_plot(shap_values, features_test, plot_type="bar", show=False)
    plt.savefig(ct.SHAP_PTH, bbox_inches='tight')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, ct.RFC_MODEL_PTH)
    joblib.dump(lrc, ct.LR_MODEL_PTH)


if __name__ == "__main__":
    df_input = import_data(ct.INPUT_DATA_PTH)
    df_input = calc_churn(df_input)
    perform_eda(df_input)
    cat_lst = ct.CATEGORY_LST
    df_input = encoder_helper(df_input, cat_lst)
    kcolumns = ct.KEEP_COLUMNS
    feat_train, feat_test, samp_train, samp_test = perform_feature_engineering(
        df_input, kcolumns)
    train_models(feat_train, feat_test, samp_train, samp_test)
    RFC_MODEL_PTH = ct.RFC_MODEL_PTH
    LR_MODEL_PTH = ct.LR_MODEL_PTH
    rfc_model = joblib.load(RFC_MODEL_PTH)
    lr_model = joblib.load(LR_MODEL_PTH)
    y_train_prf = rfc_model.predict(feat_train)
    y_train_plr = lr_model.predict(feat_train)
    y_test_prf = rfc_model.predict(feat_test)
    y_test_plr = lr_model.predict(feat_test)
    rf_classification_report_image(
        samp_train, samp_test, y_train_prf, y_test_prf)
    lr_classification_report_image(
        samp_train, samp_test, y_train_plr, y_test_plr)
    OUTPUT_PTH = ct.FEATURE_PTH
    feat = df_input[kcolumns]
    feature_importance_plot(rfc_model, feat, OUTPUT_PTH)
