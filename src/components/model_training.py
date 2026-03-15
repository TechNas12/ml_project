import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.metrics import(
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainingConfig:
    """
    Configuration container for model training artifacts.

    This dataclass defines the file paths used during the model
    training pipeline. It primarily stores the location where the
    trained model object will be serialized and saved.

    Attributes
    ----------
    trained_model_path : str
        File path where the best trained model will be stored as a
        serialized artifacts (pickle/dill format). By default, the
        model is saved inside the `artifacts/` directory as `model.pkl`.
    """
    trained_model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    """
    Handles training, evaluation, and persistence of machine learning models.

    This class orchestrates the model training stage of the machine learning
    pipeline. It evaluates multiple regression algorithms on the provided
    training and testing datasets, compares their performance using evaluation
    metrics, selects the best-performing model based on the R2 score, and
    saves the selected model as an artifacts.

    Attributes
    ----------
    model_training_config : ModelTrainingConfig
        Configuration object containing artifacts paths for storing the
        trained model.
    """
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()

    def initiate_model_training(self,training_array, test_array, preprocessor_path):
        """
    Train multiple regression models and persist the best-performing model.

    This method splits the provided training and testing arrays into
    feature matrices and target vectors, trains multiple regression models,
    evaluates their performance using several regression metrics, and
    selects the best model based on the highest test R2 score.

    If the best model's R2 score does not exceed the predefined threshold
    (0.6), an exception is raised. Otherwise, the best model is serialized
    and saved to the artifacts directory.

    Parameters
    ----------
    training_array : numpy.ndarray
        Combined training dataset containing both features and target values.
        The last column is assumed to be the target variable.

    test_array : numpy.ndarray
        Combined test dataset containing both features and target values.
        The last column is assumed to be the target variable.

    preprocessor_path : str
        File path to the saved preprocessing pipeline used during data
        transformation. (Currently unused but included for pipeline consistency.)

    Returns
    -------
    float
        R2 score of the selected best model evaluated on the test dataset.

    Raises
    ------
    CustomException
        If model training, evaluation, or artifacts saving fails, or if
        the best model does not meet the required performance threshold.
    """
        try:
            logging.info("Initiated the model training")
            logging.info("Splitting train and test data")


            X_train, y_train, X_test, y_test = (
                training_array[:,:-1],
                training_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )

            models = {
                "Random Forest" : RandomForestRegressor(),
                "Descision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "KNeighbours Regressor" : KNeighborsRegressor(),
                "XG Boost" : XGBRegressor(),
                "CatBoost Regressor" : CatBoostRegressor(verbose=False),
                "AdaBoost Regressor" : AdaBoostRegressor(),
            }

            metrics = {
                "R2 Score": r2_score,
                "Mean Absolute Error": mean_absolute_error,
                "Mean Squared Error": mean_squared_error,
                "Root Mean Squared Error": root_mean_squared_error
            }

            #ParamGrids
            params = {

                "Random Forest": {
                    "n_estimators": [50,100,200,300],
                    "max_depth": [None,5,10,20,30],
                    "min_samples_split": [2,5,10],
                    "min_samples_leaf": [1,2,4],
                    "max_features": ["sqrt","log2",None],
                    "bootstrap": [True, False]
                },

                "Descision Tree": {
                    "criterion": ["squared_error","friedman_mse","absolute_error","poisson"],
                    "splitter": ["best","random"],
                    "max_depth": [None,5,10,20,30],
                    "min_samples_split": [2,5,10],
                    "min_samples_leaf": [1,2,4],
                    "max_features": ["sqrt","log2",None]
                },

                "Gradient Boosting": {
                    "n_estimators": [50,100,200],
                    "learning_rate": [0.1,0.05,0.01],
                    "subsample": [0.6,0.7,0.8,0.9],
                    "max_depth": [3,5,7],
                    "min_samples_split": [2,5],
                    "min_samples_leaf": [1,2],
                    "max_features": ["sqrt","log2",None]
                },

                "Linear Regression": {},

                "KNeighbours Regressor": {
                    "n_neighbors": [3,5,7,9,11],
                    "weights": ["uniform","distance"],
                    "algorithm": ["auto","ball_tree","kd_tree","brute"],
                    "leaf_size": [20,30,40],
                    "p": [1,2]
                },

                "XG Boost": {
                    "n_estimators": [50,100,200],
                    "learning_rate": [0.1,0.05,0.01],
                    "max_depth": [3,5,7],
                    "subsample": [0.7,0.8,0.9],
                    "colsample_bytree": [0.7,0.8,0.9],
                    "gamma": [0,0.1,0.2],
                    "reg_alpha": [0,0.1,1],
                    "reg_lambda": [1,1.5,2]
                },

                "CatBoost Regressor": {
                    "depth": [6,8,10],
                    "learning_rate": [0.01,0.05,0.1],
                    "iterations": [100,200,300],
                    "l2_leaf_reg": [1,3,5,7],
                    "border_count": [32,64,128],
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50,100,200],
                    "learning_rate": [1,0.1,0.05,0.01],
                    "loss": ["linear","square","exponential"]
                }

            }

            logging.info("Started model evaluation ...")

            model_report, trained_models = evaluate_model(X_train = X_train, 
                                          y_train =y_train,
                                          X_test = X_test, 
                                          y_test =y_test,
                                          models=models,
                                          metrics=metrics,
                                          params=params)
            
            best_model_name = max(
                model_report,
                key = lambda model : model_report[model]['R2 Score'][1]
            )
            best_model_score = model_report[best_model_name]['R2 Score'][1]

            best_model_report = [best_model_name, best_model_score]
            best_model = trained_models[best_model_name]

            if best_model_report[1] <= 0.6:
                logging.error("Models do not exceed the threshold value.")
                raise CustomException("Model Score less than the threshold", sys)

            logging.info(f"Best model found || Name: {best_model_report[0]} | R2 Score: {best_model_report[1]}")

            logging.info("Initiated save function the best model picke file.")
            save_object(
                self.model_training_config.trained_model_path,
                obj=best_model
            )

            prediction = best_model.predict(X_test)
            score = r2_score(y_test, prediction)

            best_report = {
                "Best Model Name" : best_model_name,
                "Best Model Score" : best_model_score
            }
            return best_report

        except Exception as e:
            raise CustomException(e,sys)