import os, sys, dill, json
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):

    """
    Serialize and save a Python object to disk using dill.

    This function creates the required directory (if it does not already exist)
    and stores the provided Python object at the specified file path in
    binary format. The object is serialized using the `dill` module, which
    allows saving complex Python objects such as trained machine learning
    models, pipelines, or custom classes.

    Parameters
    ----------
    file_path : str
        The complete file path where the serialized object will be saved.
        If the directory does not exist, it will be created automatically.

    obj : Any
        The Python object to be serialized and saved. This can include
        machine learning models, preprocessing pipelines, dictionaries,
        or other Python objects.

    Raises
    ------
    CustomException
        If any error occurs during directory creation, file handling,
        or object serialization.
    """

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Saved the object successfully.")

    except Exception as e:
        raise CustomException(e,sys)
    

# Model Utils 

def save_model_report(report, file_path='artifacts/model_report.json'):
    '''
        Save the model report as a JSON artifacts.
    '''

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=4)
    
    except Exception as e:
        raise CustomException(e,sys)
    

def save_model_report_visualisation(model_report,file_path='artifacts/model_comparison.png'):
    """
    Generate and save a model comparison graph from the evaluation report.

    This function converts the nested model evaluation dictionary into a
    pandas DataFrame, extracts the test R2 scores for each model, and
    creates a bar plot comparing model performance. The resulting graph
    is saved as a 1920x1080 image inside the artifacts directory.

    Parameters
    ----------
    report : dict
        Dictionary containing model evaluation results.

        Example structure:
        {
            "Random Forest": {
                "R2 Score": [train_score, test_score],
                "Mean Absolute Error": [train_score, test_score]
            }
        }

    file_path : str, optional
        Destination path where the graph will be saved.
        Default is "artifacts/model_comparison.png".

    Raises
    ------
    CustomException
        If any error occurs during dataframe creation, plotting,
        or file saving.
    """

    try:
        logging.info("Converting model report dictionary to dataframe")

        df = (
            pd.DataFrame(model_report)
            .T
            .stack()
            .apply(pd.Series)
            .reset_index()
        )

        df.columns = ["Model", "Metric", "Train Score", "Test Score"]

        # Only use R2 for visualization
        r2_df = df[df["Metric"] == "R2 Score"]

        # Convert to long format for seaborn
        plot_df = r2_df.melt(
            id_vars="Model",
            value_vars=["Train Score", "Test Score"],
            var_name="Dataset",
            value_name="R2 Score"
        )

        logging.info("Creating grouped bar chart")

        plt.figure(figsize=(19.2, 10.8))

        sns.barplot(
            data=plot_df,
            x="Model",
            y="R2 Score",
            hue="Dataset"
        )

        plt.title("Model Comparison (R2 Score)", fontsize=18)
        plt.xlabel("Model")
        plt.ylabel("R2 Score")
        plt.xticks(rotation=45, fontsize=12)
        plt.legend(title="Dataset")
        plt.tight_layout()

        logging.info("Saving visualization artifacts")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        plt.savefig(file_path)
        plt.close()

        logging.info(f"Model comparison visualization saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, metrics, save_report=True, save_graph=True):
    """
    Train multiple machine learning models and evaluate them using provided metrics.

    This function fits each model on the training data, generates predictions for both
    training and test datasets, and computes evaluation metrics for each model. The
    results are organized into a structured report dictionary.

    Parameters
    ----------
    X_train : array-like or pandas.DataFrame
        Training feature dataset used to fit the models.

    y_train : array-like or pandas.Series
        Target values corresponding to the training dataset.

    X_test : array-like or pandas.DataFrame
        Testing feature dataset used for evaluating model performance.

    y_test : array-like or pandas.Series
        True target values for the testing dataset.

    models : dict
        Dictionary containing model names as keys and instantiated machine learning
        model objects as values.

        Example:
        {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor()
        }

    metrics : dict
        Dictionary containing metric names as keys and metric functions as values.

        Example:
        {
            "Mean Squared Error": mean_squared_error,
            "R2 Score": r2_score
        }

    report : bool
        Boolean Value to determine wheter report in form of JSON to be saved in the artifacts folder.

    graph : bool
        Boolean Value to determine wheter rcomparison graph derived from the reports should be saved in artifacts folder or not.    

    Returns
    -------
    dict
        A nested dictionary containing evaluation results for each model.

        Structure:
        {
            "Model Name": {
                "Metric Name": [train_score, test_score]
            }
        }

    Raises
    ------
    CustomException
        If any error occurs during model training or evaluation.
    """
    try:
        report = {}
        trained_models = {}
        for model_name, model in models.items():

            logging.info(f"Training model: {model_name}")

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            model_metrics = {}

            for metric_name, metric in metrics.items():

                logging.info(f"Evaluating {model_name} using {metric_name}")

                train_score = metric(y_train, y_train_pred)
                test_score = metric(y_test, y_test_pred)

                model_metrics[metric_name] = [train_score, test_score]

            report[model_name] = model_metrics
            trained_models[model_name] = model

        # Save artifacts AFTER evaluation
        if save_report:
            save_model_report(report, "artifacts/model_report.json")

        if save_graph:
            save_model_report_visualisation(report, "artifacts/model_comparison.png")

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)