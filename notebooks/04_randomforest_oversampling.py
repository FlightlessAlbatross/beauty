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

# Configuration variables
from rural_beauty.config import models_dir
import importlib

module_name = "rural_beauty.config"
para_outcome = 'beauty'
para_type = 'randomforest'
sugar = '221124_oversampled'  # Random identifier
country = 'DE'

model_basename = f"{country}_{para_outcome}_{para_type}_{sugar}"
model_folder = models_dir / model_basename
os.makedirs(model_folder, exist_ok=True)

img_model_structure = model_folder / "structure.png"
img_confusion = model_folder / "confusion_matrix.png"

# Load configuration dynamically
config_module = importlib.import_module(module_name)
feature_paths = getattr(config_module, f"feature_paths_{country}")
predictors_path = getattr(config_module, f"predictors_{country}")
outcome_path = getattr(config_module, f"outcome_{country}")

# Load datasets
with open(feature_paths, "r") as f:
    features = json.load(f)

predictors_all = pd.read_csv(predictors_path, sep=",", index_col=False, na_values=-99)
outcome = pd.read_csv(outcome_path, sep=",", index_col=False, na_values=-99)

# Clean predictors and outcome data
predictors = predictors_all.drop(columns=['hemero_1'], errors='ignore')
outcome = outcome[para_outcome].to_frame()

X    = predictors
Y    = outcome
feat = predictors.columns.values


# Ensure X and Y have aligned indices
X, Y = X.align(Y, axis=0)
# Identify rows with NA in X or invalid values in Y
na_indices_X = X.isna().any(axis=1) # Rows with any NA values in X
na_indices_Y = Y.apply(lambda x: np.any(pd.isna(x)) or np.any((x < 1) | (x > 9)), axis=1) # Rows with NA or invalid values in Y

# Combine the indices to drop
to_drop = na_indices_X | na_indices_Y
# Drop rows from X and Y based on the combined mask
X = X[~to_drop]
Y = Y[~to_drop]

Y[para_outcome] = Y[para_outcome]/2
Y[para_outcome] = Y[para_outcome].clip(lower=1, upper=5).round(0)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y[para_outcome], test_size=0.5, random_state=2024, stratify=Y)

from imblearn.over_sampling import RandomOverSampler
oversampler = RandomOverSampler(random_state=2024)
X_train_resampled, Y_train_resampled = oversampler.fit_resample(X_train, Y_train)


# Evaluate model and calculate metrics Train Decision Tree modelY_pred_test = model.predict(X_test)
model = RandomForestClassifier(max_depth=3, random_state=2024)
model.fit(X_train_resampled, Y_train_resampled)

#predict test set to calculate accurarcy scores. 
Y_pred_test = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred_test)
f1 = f1_score(Y_test, Y_pred_test, average="weighted")
kendall_tau, _ = kendalltau(Y_test, Y_pred_test)

# save model for predictions later (scripts 05_...)
joblib.dump(model, model_folder / 'model.pkl')

# save csv with sigificant features and their coeficients (if applicable; if not all are saved instead)
significant_coefs_path = model_folder / "significant_coefs.csv"

feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({ 'Feature': X_train.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
feature_importance_df.to_csv(significant_coefs_path, index = False)



# Plot and save confusion matrix with metrics
cm = confusion_matrix(Y_test, Y_pred_test, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix with Metrics")

# Adjust annotation placement to avoid overlap
plt.gca().text(
    1.2, 0.5,  # Position the text outside the plot (to the right)
    f"Accuracy: {accuracy:.2f}\nF1 Score: {f1:.2f}\nKendall's Tau: {kendall_tau:.2f}",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='center',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig(img_confusion, dpi=300)  # Save with high resolution
plt.close()

print(f"Model structure saved to: {img_model_structure}")
print(f"Confusion matrix saved to: {img_confusion}")

print(f"Model Accuracy: {accuracy:.2f}")
print(f"Model F1: {f1:.2f}")
print(f"Model Kendall's Tau: {kendall_tau:.2f}")


train_accuracies = [estimator.score(X_train_resampled.to_numpy(), Y_train_resampled.to_numpy()) for estimator in model.estimators_]
best_tree_idx = train_accuracies.index(max(train_accuracies))

# Plot and save the Decision Tree structure
plt.figure(figsize=(20, 15)) # Large figure size
plot_tree(
	model.estimators_[best_tree_idx],
	feature_names=X.columns,
	class_names=[str(cls) for cls in sorted(Y[para_outcome].unique())],
	filled=True, fontsize=10 # Increase font size 
)

plt.title("Decision Tree Structure")
plt.savefig(img_model_structure, dpi=300) # High DPI for better clarity
plt.close()
