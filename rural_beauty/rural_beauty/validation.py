import pandas as pd
import numpy as np
import joblib
import os
import json
from xgboost import XGBClassifier
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from scipy.stats import kendalltau
import matplotlib.pyplot as plt

import argparse    # to make the script accept arguments. 


# Configuration variables
from rural_beauty.config import models_dir
from rural_beauty.config import get_extracted_points_paths


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




def get_out_of_country_validation (model_folder):
    model_folder = Path(model_folder)

    module_name = "rural_beauty.config"

    # these are defined by the trained model. 
    model_basename = os.path.basename(model_folder)
    parts = model_basename.split("__")
    country, target_variable, sampling_method, model_class, class_balance, sugar = parts
    number_classes = int(sugar[0])

    # these define which points we test with. They should be whichever the model is not
    validation_area = "UK" if country == "DE" else "DE"
    validation_variable = "scenic" if country == "DE" else 'beauty'
    validation_sampling = 'pooled_pixels_all_points' if country == "DE" else 'random_pixels'

    # define output paths
    model_path               = model_folder / 'model.pkl'
    img_confusion_path       = model_folder / f"confusion_matrix_outofcountry_{validation_area}.png"

    predictors_path, outcome_path, _, _ = get_extracted_points_paths(validation_area, validation_variable, validation_sampling)

    # load model
    model = joblib.load(model_path)


    # load data for predictions
    predictors_all = pd.read_csv(predictors_path, sep=",", index_col=False, na_values=-99)
    outcome = pd.read_csv(outcome_path, sep=",", index_col=False, na_values=-99)
    ## hemerobie is not avaiable on EU level, so we cannot use it for predictions, so we drop them from training. 
    X = predictors_all.drop(columns=['hemero_1'], errors='ignore')
    predictor_names = X.columns
    Y = outcome 
    X, Y = X.align(Y, axis=0)
    na_indices_X = X.isna().any(axis=1) # Rows with any NA values in X
    na_indices_Y = Y.apply(lambda x: np.any(pd.isna(x)) or np.any((x < 0) | (x > 10)), axis=1) # Rows with NA or invalid values in Y

    ### Combine the indices to drop
    to_drop = na_indices_X | na_indices_Y
    ### Drop rows from X and Y based on the combined mask
    X = X[~to_drop]
    Y = Y[~to_drop]

    Y['value'] = Y['value'] - Y['value'].min()
    squeeze_factor = number_classes / Y['value'].max()
    Y['value'] = Y['value']*squeeze_factor
    Y = Y.round(0).clip(lower=0, upper=number_classes-1).astype(int)


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y['value'], test_size=0.2, random_state=2024, stratify=Y)

    Y_pred_test = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred_test)
    f1 = f1_score(Y_test, Y_pred_test, average="weighted")
    kendall_tau, _ = kendalltau(Y_test, Y_pred_test)

    model_info = f"Model: {model_basename}, Accuracy: {accuracy:.2f}, F1: {f1:.2f}, Kendalls Tau: {kendall_tau:.2f}"
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Model F1: {f1:.2f}")
    print(f"Model Kendall's Tau: {kendall_tau:.2f}")

    confusion_matrix_figure_todisc(output_path = img_confusion_path,
                                    Y_test = Y_test,
                                    Y_pred_test = Y_pred_test,
                                    model = model,
                                    model_basename = model_basename,
                                    accuracy = accuracy, f1 = f1, kendall_tau = kendall_tau)
    
    return (accuracy, f1, kendall_tau)

def append_model_info_to_file(file_path, model_info):
    with open(file_path, "a") as file:  # Open in append mode
        file.write(model_info + "\n")  # Append a newline after the info

def main(models_folder):
    models_folder = Path(models_folder)
    logfile_path  = models_folder / "validation_logfile.txt"

    for folder in models_folder.iterdir():
        if not folder.is_dir():  # Check if the path is a directory
            continue

        # Get the basename of the folder
        model_basename = folder.name  # Using Pathlib: `folder.name` gives the basename directly

        # Split the basename by "__"
        parts = model_basename.split("__")
        
        # Check if the basename has the expected structure
        if len(parts) != 6:
            print(f"Skipping {model_basename}: Unexpected structure.")
            continue

        # Unpack the parts
        country, target_variable, sampling_method, model_class, class_balance, sugar = parts
        number_classes = int(sugar[0])


        if country != "DE":
            continue

        accuracy, f1, kendall_tau = get_out_of_country_validation(folder)
        model_info = f"Model: {model_basename}, Accuracy: {accuracy:.2f}, F1: {f1:.2f}, Kendalls Tau: {kendall_tau:.2f}"

        append_model_info_to_file(logfile_path, model_info)







if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Script to calculate out of country accuracy")
    
    # Add arguments
    parser.add_argument(
        "models_folder",
        type=str,
        help="example: data/models/DE__unique__random_pixels__XGB__asis__7_271124"
    )

    
    # Parse arguments
    args = parser.parse_args()
    
    # Pass arguments to main function
    main(args.models_folder)

    