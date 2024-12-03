import pandas as pd
import numpy as np
import os
import json

# various ML models. 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from xgboost import XGBClassifier
from sklearn import svm

# in and export ML models to disc. 
import joblib

# ML utilities
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# selection criteria
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from scipy.stats import kendalltau

# hyperparamter tuning
from sklearn.model_selection import GridSearchCV

# to make the script accept arguments. 
import argparse    


# dynamic filepaths
from rural_beauty.config import models_dir
from rural_beauty.config import get_extracted_points_paths

# Handles oversampling in case of unbalances classes. Which all our Ys are. 
from imblearn.over_sampling import RandomOverSampler



# todo make default values for model_class, number_classes, sampling_method
def main(country, target_variable, model_class, sampling_method, number_classes, sugar, class_balance = 'asis') -> None:
    """
    Main function for training and evaluating a machine learning model based on input parameters.

    Args:
        country (str): Country code ('DE' or 'UK').
        target_variable (str): Target variable to predict (e.g., 'scenic', 'beauty').
        model_class (str): Type of ML model to use ('XGB', 'RandomForestClassifier', 'DecisionTreeClassifier').
        sampling_method (str): Data sampling method ('all_pixels', 'random_pixels', etc.).
        number_classes (int): Number of classes to classify the target variable.
        sugar (str): Unique string for differentiating between models.
        class_balance (str): Class balancing method ('oversampling', 'asis').

    Returns:
        None
    """
  
    # define parameters
    para_outcome   = target_variable
    model_basename = f"{country}__{para_outcome}__{sampling_method}__{model_class}__{class_balance}__{sugar}" # instead use something like "__".join(**kargs)
    model_folder   = models_dir / model_basename

    # define output paths
    model_path               = model_folder / 'model.pkl'
    img_model_structure_path = model_folder / "structure.png"
    img_confusion_path       = model_folder / "confusion_matrix.png"
    significant_coefs_path   = model_folder / "significant_coefs.csv"
    logfile_path             = models_dir / "logfile.txt"

    # get input paths
    predictors_path, outcome_path, _, _ = get_extracted_points_paths(country, target_variable, sampling_method)

    # load data
    predictors_all = pd.read_csv(predictors_path, sep=",", index_col=False, na_values=-99)
    outcome = pd.read_csv(outcome_path, sep=",", index_col=False, na_values=-99)

    ## hemerobie is not avaiable on EU level, so we cannot use it for predictions, so we drop them from training. 
    predictors = predictors_all.drop(columns=['hemero_1'], errors='ignore')

    # drop NAs
    X, Y = handle_na(predictors, outcome)

    ## Squeeze the outcome classes down to the specified level of class_balancing. Some of our classes are almost empty, this helps that. 
    Y = sqeeze_Y_classes(Y, number_classes)

    # test/train split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y['value'], test_size=0.8, random_state=2024, stratify=Y)


    # over/under sampling or keep it as is. 
    X_train_balanced, Y_train_balanced = cases_resample_data(X_train, Y_train, class_balance)

    # select model
    model = cases_assign_model_class(model_class)

    hyperparameter_training = True
    if hyperparameter_training:
        results = hyperparameter_tuning(model_class, X_train, Y_train)
        return results
    

    # train model
    model.fit(X_train_balanced, Y_train_balanced)

    # test metrics 
    ## predictions for test metrics 
    Y_pred_test = model.predict(X_test)
    ## calculate test metrics
    accuracy       = accuracy_score(Y_test, Y_pred_test)
    f1             = f1_score      (Y_test, Y_pred_test, average="weighted")
    kendall_tau, _ = kendalltau    (Y_test, Y_pred_test)

    # This string gets written to a logfile later. 
    model_info = f"Model: {model_basename}, Accuracy: {accuracy:.2f}, F1: {f1:.2f}, Kendalls Tau: {kendall_tau:.2f}"
    # This one to the console.
    print(f"Model Accuracy:      {accuracy:.2f}")
    print(f"Model F1:            {f1:.2f}")
    print(f"Model Kendall's Tau: {kendall_tau:.2f}")

    # save outputs
    os.makedirs(model_folder, exist_ok=True)

    ## save model for predictions later (scripts 05_...)
    joblib.dump(model, model_path)
    ## save csv with sigificant features and their coeficients (if applicable; if not all are saved instead)
    # TODO: make the coeffiecients work for all model types
    # coefs = get_model_parameters_RF(model, predictor_names)
    # coefs.to_csv(significant_coefs_path, index = False)
    append_model_info_to_file(logfile_path, model_info)


    ## create and save Confusion matrix
    confusion_matrix_figure_todisc(output_path = img_confusion_path,
                                   Y_test = Y_test,
                                   Y_pred_test = Y_pred_test,
                                   model = model,
                                   model_basename = model_basename,
                                   accuracy = accuracy, f1 = f1, kendall_tau = kendall_tau)

    # TODO make work for all models    
    # tree_structure_RF_best_accuracy_tree(output_path = img_model_structure_path,
    #                                      X_train = X_train,
    #                                      Y_train = Y_train,
    #                                      model = model,
    #                                      model_basename = model_basename)



def hyperparameter_tuning(model_class, X, y, cv=5):
    """
    Perform hyperparameter cross-validation tuning for XGBoost, RandomForest, or DecisionTree.

    Args:
        model_class (str): Model type ('XGB', 'RF', or 'Tree').
        X (pd.DataFrame or np.array): Feature matrix.
        y (pd.Series or np.array): Target vector.
        cv (int): Number of cross-validation folds. Defaults to 5.

    Returns:
        dict: Best hyperparameters and their corresponding model.
    """
    # Define model and hyperparameter grid based on model class
    if model_class == "XGB":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": [50, 100, 200],
            "subsample": [0.8, 1.0],
        }
    elif model_class == "RF":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        }
    elif model_class == "Tree":
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    else:
        raise ValueError("Invalid model_class. Choose from 'XGB', 'RF', or 'Tree'.")

    # Perform grid search cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        verbose=2,
        n_jobs=-1,  # Use all available cores
    )
    grid_search.fit(X, y)

    # Return the best parameters and best model
    return {
        "best_params": grid_search.best_params_,
        "best_model": grid_search.best_estimator_,
        "best_score": grid_search.best_score_,
    }



def get_model_parameters_RF(model, predictor_names):
    feature_importances = model.feature_importances_
    return pd.DataFrame({ 'Feature': predictor_names, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

   


# Plot and save confusion matrix with metrics
def confusion_matrix_figure_todisc(output_path, Y_test, Y_pred_test, model, model_basename, accuracy, f1, kendall_tau) -> None: 
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    cm = confusion_matrix(Y_test, Y_pred_test, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix with Metrics")

    # Metrics annotation outside the plot
    plt.gca().text(
        1.2, 0.5,  # Position the text outside the plot (to the right)
        f"Accuracy: {accuracy:.2f}\nF1 Score: {f1:.2f}\nKendall's Tau: {kendall_tau:.2f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='center',
        horizontalalignment='left',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # Add model basename as an annotation at the bottom of the figure
    plt.figtext(
        0.5, 0.01,  # Centered at the bottom of the figure
        f"Model: {model_basename}",
        fontsize=9,
        ha="center",  # Horizontal alignment
        va="center",  # Vertical alignment
        alpha=0.7
    )

    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(output_path, dpi=300)  # Save with high resolution
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def tree_structure_RF_best_accuracy_tree(output_path, X_train, Y_train, model, model_basename):
    train_accuracies = [estimator.score(X_train.to_numpy(), Y_train.to_numpy()) for estimator in model.estimators_]
    best_tree_idx = train_accuracies.index(max(train_accuracies))

    # Plot and save the Decision Tree structure
    plt.figure(figsize=(20, 15)) # Large figure size
    plot_tree(
        model.estimators_[best_tree_idx],
        feature_names=X_train.columns,
        class_names=[str(cls) for cls in sorted(Y_train.unique())],
        filled=True, fontsize=10 # Increase font size 
    )

    # Add model basename as an annotation at the bottom of the figure
    plt.figtext(
        0.5, 0.01,  # Centered at the bottom of the figure
        f"Model: {model_basename}",
        fontsize=9,
        ha="center",  # Horizontal alignment
        va="center",  # Vertical alignment
        alpha=0.7
    )

    plt.title("Decision Tree Structure of the highest accuracy tree.")
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(output_path, dpi=300) # High DPI for better clarity
    plt.close()
    print(f"Model structure saved to: {output_path}")


def handle_na(X, Y):
    """
    Handles missing values and invalid entries in the predictors and outcome datasets.

    Args:
        X (pd.DataFrame): Predictor variables.
        Y (pd.DataFrame): Outcome variable.

    Returns:
        tuple: Cleaned predictors (X) and outcome (Y) with matching indices.
    """
    ## Ensure X and Y have aligned indices
    X, Y = X.align(Y, axis=0)

    ## NAs
    ### Identify rows with NA in X or invalid values in Y
    na_indices_X = X.isna().any(axis=1) # Rows with any NA values in X
    na_indices_Y = Y.apply(lambda x: np.any(pd.isna(x)) or np.any((x < 0) | (x > 10)), axis=1) # Rows with NA or invalid values in Y

    ### Combine the indices to drop
    to_drop = na_indices_X | na_indices_Y
    ### Drop rows from X and Y based on the combined mask
    X = X[~to_drop]
    Y = Y[~to_drop]
    return X,Y


def sqeeze_Y_classes (Y, number_classes):
    # XGB needs the lowest class to be 0
    Y['value'] = Y['value'] - Y['value'].min()

    squeeze_factor = number_classes / Y['value'].max()
    Y['value'] = Y['value']*squeeze_factor
    
    # print(f"DEBUG: miny {Y.min()}, maxy {Y.max()}")
    Y = Y.round(0).clip(lower=0, upper=number_classes-1).astype(int)
    return Y

def cases_resample_data(X_train, Y_train, class_balance):
    """
    Resamples training data based on the specified class balancing method.

    Args:
        X_train (pd.DataFrame): Training predictors.
        Y_train (pd.Series): Training labels.
        class_balance (str): Class balancing method ('oversampling', 'asis').

    Returns:
        tuple: Resampled predictors (X_train_balanced) and labels (Y_train_balanced).
    """
    match (class_balance):
        case ("oversampling"):
            oversampler = RandomOverSampler(random_state=2024)
            X_train_balanced, Y_train_balanced = oversampler.fit_resample(X_train, Y_train)
        case ("asis"):
            X_train_balanced = X_train
            Y_train_balanced = Y_train
        case _:
            raise ValueError(f"Invalid case for: {class_balance}. Try oversampling, undersampling or asis")
        
    return X_train_balanced, Y_train_balanced

def cases_assign_model_class(model_class):
    """
    Assigns and initializes a machine learning model class based on the input string.

    Args:
        model_class (str): Type of ML model to initialize ('RandomForestClassifier', 'XGB', etc.).

    Returns:
        object: Initialized ML model.

    Raises:
        ValueError: If an unsupported model class is provided.
    """
    match (model_class):
        case ("RandomForestClassifier"):
            model = RandomForestClassifier(max_depth=3, random_state=2024)
        case ("XGB"):
            model = XGBClassifier(tree_method="hist")
        case ("DecisionTreeClassifier"):
            model = DecisionTreeClassifier(max_depth=3, random_state=2024)
        case ("LinearRegression"):
            LinearRegression()
        case _:
            raise ValueError ("NO valid model_class has been selected. Try RandomForestClassifier or XGB or SVC or DescisionTreeClassifier")
    return model


def append_model_info_to_file(file_path, model_info):
    with open(file_path, "a") as file:  # Open in append mode
        file.write(model_info + "\n")  # Append a newline after the info

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train one of various ML models on preselected data for Germany or the UK")

    parser.add_argument("country"         , type=str, choices=['DE', 'UK'], help="Country code ('DE', 'UK', or 'DEUK').")
    parser.add_argument("target_variable" , type=str, choices=['scenic', 'beauty', 'unique', 'diverse'], help="Target variable (e.g., 'beauty', 'scenic').")
    parser.add_argument("model_class"     , type=str, choices=['XGB', 'RandomForestClassifier', 'DecisionTreeClassifier'], help="Model class: RandomForestClassifier, TreeClassifier, XGB...")
    parser.add_argument("sampling_method" , type=str, choices=['all_pixels', 'random_pixels', 'pooled_pixels_all_points', 'pooled_pixels_random_points'], help="Sampling method for data extraction used (e.g., 'all_pixels', 'random_pixels').")
    parser.add_argument("number_classes"  , type=int, help="This sets the number of classes in the model. Fewer classes can be easier to predict")   
    parser.add_argument("sugar"           , type=str, help="Any unique string to differentieate between models. This will be added to the output model folder name")
    parser.add_argument("--class_balance" , type=str, default='asis', choices=['oversamping', 'asis'], help="Oversampling repeats low freqency classes, asis uses the data as is, undersampling doesn't repeat any entries, but reduces the classes with too many")
    # Get arguments from command line
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(
        country          = args.country,
        target_variable  = args.target_variable,
	    model_class      = args.model_class,
        class_balance    = args.class_balance,
        number_classes   = args.number_classes,
        sugar            = args.sugar,
        sampling_method  = args.sampling_method
    )
