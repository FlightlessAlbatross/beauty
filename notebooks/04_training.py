import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

import argparse    # to make the script accept arguments. 


# Configuration variables
from rural_beauty.config import models_dir
from rural_beauty.config import get_extracted_points_paths

from imblearn.over_sampling import RandomOverSampler


def main(country, target_variable, sampling_method, class_balance, number_classes, sugar):

    # define parameters
    para_outcome  = target_variable
    model_class   = 'randomforest' # TODO: make it a parameter
    model_basename = f"{country}__{para_outcome}__{sampling_method}__{model_class}__{class_balance}__{sugar}"
    model_folder = models_dir / model_basename
    os.makedirs(model_folder, exist_ok=True)

    # define output paths
    model_path               = model_folder / 'model.pkl'
    img_model_structure_path = model_folder / "structure.png"
    img_confusion_path       = model_folder / "confusion_matrix.png"
    significant_coefs_path   = model_folder / "significant_coefs.csv"

    # get input paths
    predictors_path, outcome_path, _, _ = get_extracted_points_paths(country, target_variable, sampling_method)

    # load data
    predictors_all = pd.read_csv(predictors_path, sep=",", index_col=False, na_values=-99)
    outcome = pd.read_csv(outcome_path, sep=",", index_col=False, na_values=-99)


    ## hemerobie is not avaiable on EU level, so we cannot use it for predictions, so we drop them from training. 
    X = predictors_all.drop(columns=['hemero_1'], errors='ignore')
    predictor_names = X.columns
    Y = outcome
    ## Ensure X and Y have aligned indices
    X, Y = X.align(Y, axis=0)

    ## why is this needed? comment out for now. TODO Delete. 
    # feat = predictors.columns.values

    ## NAs
    ### Identify rows with NA in X or invalid values in Y
    na_indices_X = X.isna().any(axis=1) # Rows with any NA values in X
    na_indices_Y = Y.apply(lambda x: np.any(pd.isna(x)) or np.any((x < 1) | (x > 9)), axis=1) # Rows with NA or invalid values in Y

    ### Combine the indices to drop
    to_drop = na_indices_X | na_indices_Y
    ### Drop rows from X and Y based on the combined mask
    X = X[~to_drop]
    Y = Y[~to_drop]

    ## Squeeze the outcome classes down to the specified level of class_balancing
    ### squeeze_factor tells us by which factor we need to devide our classes.
    squeeze_factor = number_classes / Y.max()

    Y = Y*squeeze_factor
    Y = Y.round(0).clip(lower=1, upper=number_classes).astype(int)
    
    # test/train split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y['value'], test_size=0.5, random_state=2024, stratify=Y)
    # over/under sampling

    match (class_balance):
        case ("oversampling"):
            oversampler = RandomOverSampler(random_state=2024)
            X_train_balanced, Y_train_balanced = oversampler.fit_resample(X_train, Y_train)
        case ("undersampling"):
            print("undersampling is not yet implemented")
            pass
        case ("asis"):
            X_train_balanced = X_train
            Y_train_balanced = Y_train
        case _:
            raise ValueError(f"Invalid case for: {class_balance}. Try oversampling, undersampling or asis")


    # train model
    model = RandomForestClassifier(max_depth=3, random_state=2024)
    model.fit(X_train_balanced, Y_train_balanced)

    # test metrics 
    ## predictions for test metrics 
    Y_pred_test = model.predict(X_test)
    ## calculate test metrics
    accuracy = accuracy_score(Y_test, Y_pred_test)
    f1 = f1_score(Y_test, Y_pred_test, average="weighted")
    kendall_tau, _ = kendalltau(Y_test, Y_pred_test)
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Model F1: {f1:.2f}")
    print(f"Model Kendall's Tau: {kendall_tau:.2f}")

    # save outputs
    ## save model for predictions later (scripts 05_...)
    joblib.dump(model, model_path)
    ## save csv with sigificant features and their coeficients (if applicable; if not all are saved instead)
    coefs = get_model_parameters_RF(model, predictor_names)
    coefs.to_csv(significant_coefs_path, index = False)

    ## create and save Confusion matrix
    confusion_matrix_figure_todisc(output_path = img_confusion_path,
                                   Y_test = Y_test,
                                   Y_pred_test = Y_pred_test,
                                   model = model,
                                   model_basename = model_basename,
                                   accuracy = accuracy, f1 = f1, kendall_tau = kendall_tau)
    
    tree_structure_RF_best_accuracy_tree(output_path = img_model_structure_path,
                                         X_train = X_train,
                                         Y_train = Y_train,
                                         model = model,
                                         model_basename = model_basename)




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


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Model the target variable using Random Forest. ")
    parser.add_argument("country", type=str, help="Country code ('DE', 'UK', or 'DEUK').")
    parser.add_argument("target_variable", type=str, help="Target variable (e.g., 'beauty', 'scenic').")
    parser.add_argument("sampling_method", type=str, help="Sampling method for data extraction used (e.g., 'all_pixels', 'random_pixels').")
    parser.add_argument("class_balance", type=str, default=True, help="Oversampling repeats low freqency classes, asis uses the data as is, undersampling doesn't repeat any entries, but reduces the classes with too many")
    parser.add_argument("number_classes", type=int, default = 10,  help="This sets the number of classes in the model. Fewer classes can be easier to predict")   
    parser.add_argument("sugar"         , type=str, default = "000000", help = "Any unique string to differentieate between models. This will be added to the output model folder name")
    # Get arguments from command line
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(
        country=args.country,
        target_variable=args.target_variable,
        sampling_method=args.sampling_method,
        class_balance=args.class_balance,
        number_classes = args.number_classes,
        sugar          = args.sugar
    )