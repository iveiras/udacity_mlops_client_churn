'''
Module to test the churn_library functions

Author: Iago Veiras
Date: March 2022
'''

import os
import logging
import joblib
import churn_library as cl
import constants as ct

logging.basicConfig(
    filename=f"./logs/churn_library_{time.strftime('%b_%d_%Y_%H_%M_%S')}.log",
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        input_df = import_data(ct.INPUT_DATA_PTH)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert input_df.shape[0] > 0
        assert input_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_calc_churn(calc_churn):
    '''
    test perform calc_churn function
    '''
    try:
        input_df = cl.import_data(ct.INPUT_DATA_PTH)
        input_df = calc_churn(input_df)
        logging.info("Testing calc_churn: SUCCESS")
    except KeyError as err:
        logging.error(
            "Testing calc_churn: Missing Churn variable from the dataframe")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        input_df = cl.import_data(ct.INPUT_DATA_PTH)
        input_df = cl.calc_churn(input_df)
        perform_eda(input_df)
        logging.info("Testing perform_eda: SUCCESS")
    except KeyError as err:
        logging.error(
            "Testing perform_eda: Mssing a needed variable from the dataframe")
        raise err
    except FileNotFoundError as err:
        logging.error("Testing perform_eda: Destination folder wasn't found")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        input_df = cl.import_data(ct.INPUT_DATA_PTH)
        input_df = cl.calc_churn(input_df)
        category_lst = ct.CATEGORY_LST
        assert all(cat in input_df.columns for cat in category_lst) is True
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: A varible from the list doesn't exist in the dataframe")
        raise err

    try:
        assert all(cat == 'object' for cat in input_df[category_lst].dtypes) is True
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: A variable from the list is not categorycal")
        raise err

    try:
        response = [cat + '_Churn' for cat in category_lst]
        assert len(category_lst) == len(response)
        input_df = encoder_helper(input_df, category_lst, response)
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Category list and response list don't have the same length")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        input_df = cl.import_data(ct.INPUT_DATA_PTH)
        input_df = cl.calc_churn(input_df)
        category_lst = ct.CATEGORY_LST
        input_df = cl.encoder_helper(input_df, category_lst)
        keep_columns = ct.KEEP_COLUMNS
        assert all(column in input_df.columns for column in keep_columns) is True
        perform_feature_engineering(input_df, keep_columns)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Filtered column doesn't exist in the dataframe")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        input_df = cl.import_data(ct.INPUT_DATA_PTH)
        input_df = cl.calc_churn(input_df)
        category_lst = ct.CATEGORY_LST
        input_df = cl.encoder_helper(input_df, category_lst)
        keep_columns = ct.KEEP_COLUMNS
        features_train, features_test, samples_train, samples_test = cl.perform_feature_engineering(
            input_df, keep_columns)
        assert features_train.shape[1] == features_test.shape[1]
        assert features_train.shape[0] == len(samples_train)
        assert features_test.shape[0] == len(samples_test)
        train_models(features_train, features_test, samples_train, samples_test)
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: Train and test subsets doesn't match in size")
        raise err


def test_rf_classification_report_image(rf_classification_report_image):
    '''
    test rf_classification_report_image
    '''
    try:
        input_df = cl.import_data(ct.INPUT_DATA_PTH)
        input_df = cl.calc_churn(input_df)
        category_lst = ct.CATEGORY_LST
        input_df = cl.encoder_helper(input_df, category_lst)
        keep_columns = ct.KEEP_COLUMNS
        features_train, features_test, samples_train, samples_test = cl.perform_feature_engineering(
            input_df, keep_columns)
        rfc_model_pth = ct.RFC_MODEL_PTH
        assert os.path.isfile(rfc_model_pth)
        rfc_model = joblib.load(rfc_model_pth)
        y_train_preds_rf = rfc_model.predict(features_train)
        y_test_preds_rf = rfc_model.predict(features_test)
        rf_classification_report_image(
            samples_train, samples_test, y_train_preds_rf, y_test_preds_rf)
        logging.info("Testing rf_classification_report_image: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing rf_classification_report_image: Model path is incorrect")
        raise err


def test_lr_classification_report_image(lr_classification_report_image):
    '''
    test lr_classification_report_image
    '''
    try:
        input_df = cl.import_data(ct.INPUT_DATA_PTH)
        input_df = cl.calc_churn(input_df)
        category_lst = ct.CATEGORY_LST
        input_df = cl.encoder_helper(input_df, category_lst)
        keep_columns = ct.KEEP_COLUMNS
        features_train, features_test, samples_train, samples_test = cl.perform_feature_engineering(
            input_df, keep_columns)
        lr_model_pth = ct.LR_MODEL_PTH
        assert os.path.isfile(lr_model_pth)
        lr_model = joblib.load(lr_model_pth)
        y_train_preds_lr = lr_model.predict(features_train)
        y_test_preds_lr = lr_model.predict(features_test)
        lr_classification_report_image(
            samples_train, samples_test, y_train_preds_lr, y_test_preds_lr)
        logging.info("Testing lr_classification_report_image: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing lr_classification_report_image: Model path is incorrect")
        raise err


def test_feature_importance_plot(feature_importance_plot):
    '''
    test feature_importance_plot
    '''
    try:
        input_df = cl.import_data(ct.INPUT_DATA_PTH)
        input_df = cl.calc_churn(input_df)
        category_lst = ct.CATEGORY_LST
        input_df = cl.encoder_helper(input_df, category_lst)
        keep_columns = ct.KEEP_COLUMNS
        features = input_df[keep_columns]
        model_pth = ct.RFC_MODEL_PTH
        assert os.path.isfile(model_pth)
        model = joblib.load(model_pth)
    except AssertionError as err:
        logging.error(
            "Testing feature_importance_plot: Model path is incorrect")
        raise err

    try:
        output_pth = ct.FEATURE_PTH
        output_folder = os.path.dirname(output_pth)
        assert os.path.isdir(output_folder)
        feature_importance_plot(model, features, output_pth)
        logging.info("Testing feature_importance_plot: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing feature_importance_plot: Output path folder doesn't exit")
        raise err


if __name__ == "__main__":
    test_import(cl.import_data)
    test_calc_churn(cl.calc_churn)
    test_eda(cl.perform_eda)
    test_encoder_helper(cl.encoder_helper)
    test_perform_feature_engineering(cl.perform_feature_engineering)
    test_train_models(cl.train_models)
    test_rf_classification_report_image(cl.rf_classification_report_image)
    test_lr_classification_report_image(cl.lr_classification_report_image)
    test_feature_importance_plot(cl.feature_importance_plot)
