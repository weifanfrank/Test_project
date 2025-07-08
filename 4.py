pip install tensorflow

=====

import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, confusion_matrix

# Set a fixed random seed for reproducibility
import numpy as np
np.random.seed(42)

# Load the dataset
df = pd.read_excel('/content/Cleaned_Drugs_with_Averages.xlsx')

# Selecting columns with 'Bit_' in their names
bit_columns = [col for col in df.columns if 'Bit_' in col]
X = df[bit_columns]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA transformation with 12 components
pca = PCA(n_components=2, random_state=42)  # Set random_state if PCA uses randomness
X_pca = pca.fit_transform(X_scaled)

# Target variable
y = df['average']

# Converting the target into binary based on median value
median_gfp = y.median()
y_binary = (y > median_gfp).astype(int)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train_binary, y_test_binary = train_test_split(X_pca, y_binary, test_size=0.2, random_state=42)

# Random Forest Classifier with a fixed random state
rf_classifier = RandomForestClassifier(random_state=42)

# Parameters for Grid Search
rf_parameters = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search with 5-fold cross-validation
rf_grid_search = GridSearchCV(rf_classifier, rf_parameters, cv=11, scoring='f1_macro')
rf_grid_search.fit(X_train, y_train_binary)

# Best parameters and estimator
best_rf_params = rf_grid_search.best_params_
best_rf_classifier = rf_grid_search.best_estimator_

# Predictions
y_pred_best_rf = best_rf_classifier.predict(X_test)

# Classification report
report_best_rf = classification_report(y_test_binary, y_pred_best_rf, output_dict=True)
report_best_rf_df = pd.DataFrame(report_best_rf).transpose()
print(report_best_rf_df[['precision', 'recall', 'f1-score']])

# Mean Squared Error
mse_rf = mean_squared_error(y_test_binary, y_pred_best_rf)
print("Mean Squared Error (Random Forest):", mse_rf)

# Confusion Matrix
confusion_mat = confusion_matrix(y_test_binary, y_pred_best_rf)
print("Confusion Matrix:")
print(confusion_mat)

# Validation and Test Scores
validation_score = rf_grid_search.best_score_
test_score = best_rf_classifier.score(X_test, y_test_binary)
print(f"Validation Score (F1 macro): {validation_score:.4f}")
print(f"Test Score (Accuracy): {test_score:.4f}")
# Save the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Save the PCA transformer
with open('pca_transformer.pkl', 'wb') as file:
    pickle.dump(pca, file)

# Save the best Random Forest Classifier
with open('rf_model.pkl', 'wb') as file:
    pickle.dump(best_rf_classifier, file)

=====

pip install deepchem

=====

pip install --upgrade scipy

=====

Change "featurizer_size" here to have different number of features

-----

import pandas as pd
import deepchem as dc
import numpy as np
import logging

def load_data(file_path):
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        logging.error(f"Error loading Excel file: {e}")
        raise

def validate_smiles(smiles_list):
    return smiles_list

def featurize_smiles(smiles, featurizer):
    try:
        return featurizer.featurize(smiles)
    except Exception as e:
        logging.warning(f"Failed to featurize {smiles}: {e}")
        return None

def main(file_path, output_file, featurizer_size=96):
    logging.basicConfig(level=logging.INFO)
    data = load_data(file_path)
    smiles = validate_smiles(data['Unnamed: 4'][1:].astype(str))

    featurizer = dc.feat.CircularFingerprint(size=featurizer_size)

    features = [featurize_smiles(smile, featurizer) for smile in smiles if smile is not None]

    features_2d = np.array([f for f in features if f is not None]).squeeze()

    features_df = pd.DataFrame(features_2d)
    features_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    file_path = '/content/Drugs with isomeric smile.xlsx'
    output_file = 'featurized_data.csv'
    main(file_path, output_file)

=====

pip install xgboost

=====

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

features_df = pd.read_csv(output_file)
target_data = load_data('/content/Drugs_target.xlsx')
target = target_data['average']

threshold = np.median(target)
target_binary = (target > threshold).astype(int)

X_train, X_test, y_train, y_test = train_test_split(features_df, target_binary, test_size=0.2, random_state=42)

xgb_classifier = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    reg_alpha=0,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

xgb_classifier.fit(X_train, y_train)

y_pred = xgb_classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

=====

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(features_df, target_binary, test_size=0.2, random_state=42)

max_depths = [2, 3, 4, 5, 6, 7, 8]
learning_rates = [0.0001, 0.005, 0.001, 0.05, 0.04, 0.03, 0.02, 0.01, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
n_estimators_list = [50, 80, 100, 120, 150, 180, 200]
min_split_losses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
min_child_weights= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
subsamples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
max_delta_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

best_accuracy = 0
best_params = {}

for max_depth in max_depths:
    for learning_rate in learning_rates:
        for n_estimators in n_estimators_list:
            for min_split_loss in min_split_losses:
                for min_child_weight in min_child_weights:
                    for subsample in subsamples:
                        for max_delta_step in max_delta_steps:
                            xgb_classifier = xgb.XGBClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                min_split_loss=min_split_loss,
                                min_child_weight=min_child_weight,
                                max_delta_step=max_delta_step,
                                subsample=subsample,
                                colsample_bytree=0.8,
                                reg_lambda=1,
                                reg_alpha=0,
                                objective='binary:logistic',
                                random_state=42
                            )
                            xgb_classifier.fit(X_train, y_train)

                            y_pred = xgb_classifier.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)

                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_params = {
                                    'max_depth': max_depth,
                                    'learning_rate': learning_rate,
                                    'n_estimators': n_estimators,
                                    'min_split_loss': min_split_loss,
                                    'min_child_weight': min_child_weight,
                                    'subsample': subsample,
                                    'max_delta_step':max_delta_step
                                }
                                print(f"New best parameters: {best_params} with accuracy: {best_accuracy}")

print(f"Best parameters: {best_params} with accuracy: {best_accuracy}")


=====

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(features_df, target_binary, test_size=0.2, random_state=42)

max_depths = [4, 5, 6, 7, 8]
learning_rates = [0.0001, 0.005, 0.001, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.1, 0.2, 0.5]
n_estimators_list = [50, 80, 100, 120, 150, 180, 200, 220, 250]


best_accuracy = 0
best_params = {}

for max_depth in max_depths:
    for learning_rate in learning_rates:
        for n_estimators in n_estimators_list:
            xgb_classifier = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1,
                reg_alpha=0,
                objective='binary:logistic',
                random_state=42
            )
            xgb_classifier.fit(X_train, y_train)

            y_pred = xgb_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'n_estimators': n_estimators,
                }
                print(f"New best parameters: {best_params} with accuracy: {best_accuracy}")

print(f"Best parameters: {best_params} with accuracy: {best_accuracy}")

=====

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score



X_train, X_pool, y_train, y_pool = train_test_split(features_df, target_binary, test_size=0.5, random_state=42)


X_pool, X_test, y_pool, y_test = train_test_split(X_pool, y_pool, test_size=0.2, random_state=42)

best_accuracy = 0
best_params = {}


max_depths = [1,2,3, 4, 5, 6, 7, 8]
learning_rates = [0.0001, 0.005, 0.001, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.1, 0.2, 0.5]
n_estimators_list = [10,20,30,40,50, 60,70,80, 90,100, 110, 120, 125,130,150,160,170, 180, 190,200, 210,220, 230,240,250]


n_active_learning_iterations = 10
n_queries_per_iteration = 10

for iteration in range(n_active_learning_iterations):
    print(f"Active learning iteration: {iteration + 1}")

    for max_depth in max_depths:
        for learning_rate in learning_rates:
            for n_estimators in n_estimators_list:
                xgb_classifier = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1,
                    reg_alpha=0,
                    objective='binary:logistic',
                    random_state=42
                )

                xgb_classifier.fit(X_train, y_train)
                y_pred = xgb_classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'n_estimators': n_estimators,
                    }
                    print(f"New best parameters: {best_params} with accuracy: {best_accuracy}")


    if len(X_pool) > n_queries_per_iteration:
        indices = np.random.choice(range(len(X_pool)), n_queries_per_iteration, replace=False)
        X_queried = X_pool.iloc[indices]
        y_queried = y_pool.iloc[indices]


        X_pool = X_pool.drop(X_pool.index[indices])
        y_pool = y_pool.drop(y_pool.index[indices])


        X_train = pd.concat([X_train, X_queried])
        y_train = pd.concat([y_train, y_queried])
    else:
        print("Not enough samples in pool to query.")
        break

    print(f"End of iteration {iteration + 1}. Best accuracy so far: {best_accuracy}")

print(f"Final best parameters: {best_params} with accuracy: {best_accuracy}")

=====

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming features_df and target_binary are defined elsewhere in your notebook
X_train, X_pool, y_train, y_pool = train_test_split(features_df, target_binary, test_size=0.5, random_state=42)
X_pool, X_test, y_pool, y_test = train_test_split(X_pool, y_pool, test_size=0.2, random_state=42)

best_accuracy = 0
best_params = {}

max_depths = [2]
learning_rates = [0.08]
n_estimators_list = [30]

n_active_learning_iterations = 12
n_queries_per_iteration = 12

for iteration in range(n_active_learning_iterations):
    print(f"Active learning iteration: {iteration + 1}")

    for max_depth in max_depths:
        for learning_rate in learning_rates:
            for n_estimators in n_estimators_list:
                xgb_classifier = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1,
                    reg_alpha=0,
                    objective='binary:logistic',
                    random_state=42
                )

                xgb_classifier.fit(X_train, y_train)
                y_pred = xgb_classifier.predict(X_test)

                # Calculate and print accuracy, precision, recall, and F1 score
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary')
                recall = recall_score(y_test, y_pred, average='binary')
                f1 = f1_score(y_test, y_pred, average='binary')

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'n_estimators': n_estimators,
                    }
                    print(f"New best parameters: {best_params} with accuracy: {best_accuracy}, precision: {precision}, recall: {recall}, F1: {f1}")

    # Check if there are enough samples in the pool to perform the next query
    n_samples_left = len(X_pool)
    if n_samples_left >= n_queries_per_iteration:
        query_size = n_queries_per_iteration
    else:
        query_size = n_samples_left  # If fewer than n_queries_per_iteration, use all the remaining samples

    # If there are samples left to query, perform the active learning step
    if query_size > 0:
        indices = np.random.choice(range(n_samples_left), query_size, replace=False)
        X_queried = X_pool.iloc[indices]
        y_queried = y_pool.iloc[indices]

        X_pool = X_pool.drop(X_pool.index[indices])
        y_pool = y_pool.drop(y_pool.index[indices])

        X_train = pd.concat([X_train, X_queried])
        y_train = pd.concat([y_train, y_queried])
    else:
        print("No more samples in pool to query.")
        break

    print(f"End of iteration {iteration + 1}. Best accuracy so far: {best_accuracy}")

print(f"Final best parameters: {best_params} with accuracy: {best_accuracy}")


=====

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)

# Assuming features_df and target_binary are defined elsewhere in your notebook
n_active_learning_iterations = 3
n_queries_per_iteration = 48
total_data_size = len(features_df)

# Initialize training set to None
X_train = None
y_train = None
X_queried = None
y_queried = None

# Initialize test set
X_train, X_pool, y_train, y_pool = train_test_split(features_df, target_binary, test_size=0.5, random_state=42)
X_pool, X_test, y_pool, y_test = train_test_split(X_pool, y_pool, test_size=0.2, random_state=42)

for iteration in range(n_active_learning_iterations):
    print(f"Active learning iteration: {iteration + 1}")

    # If it's not the first iteration, concatenate queried samples with training set
    if X_train is not None:
        X_train = pd.concat([X_train, X_queried], ignore_index=True)
        y_train = pd.concat([y_train, y_queried], ignore_index=True)
    else:
        # If it's the first iteration, set training set to the initial training set
        X_train, X_pool, y_train, y_pool = train_test_split(features_df, target_binary, test_size=0.5, random_state=42)

    best_accuracy = 0
    best_params = {}
    best_confusion_matrix_test = None
    best_confusion_matrix_train = None
    best_accuracy_train = 0

    max_depths = [1]
    learning_rates = [0.05]
    n_estimators_list = [200]

    for max_depth in max_depths:
        for learning_rate in learning_rates:
            for n_estimators in n_estimators_list:
                xgb_classifier = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1,
                    reg_alpha=0,
                    objective='binary:logistic',
                    random_state=42
                )

                xgb_classifier.fit(X_train, y_train)

                y_pred_test = xgb_classifier.predict(X_test)
                y_pred_train = xgb_classifier.predict(X_train)

                accuracy_test = accuracy_score(y_test, y_pred_test)
                precision_test = precision_score(y_test, y_pred_test)
                recall_test = recall_score(y_test, y_pred_test)
                f1_test = f1_score(y_test, y_pred_test)

                accuracy_train = accuracy_score(y_train, y_pred_train)
                precision_train = precision_score(y_train, y_pred_train)
                recall_train = recall_score(y_train, y_pred_train)
                f1_train = f1_score(y_train, y_pred_train)

                if accuracy_test > best_accuracy:
                    best_accuracy = accuracy_test
                    best_accuracy_train = accuracy_train
                    best_params = {
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'n_estimators': n_estimators,
                    }
                    best_confusion_matrix_test = confusion_matrix(y_test, y_pred_test)
                    best_confusion_matrix_train = confusion_matrix(y_train, y_pred_train)

    print(f"Iteration: {iteration + 1} - Best parameters: {best_params} with test accuracy: {best_accuracy}, training accuracy: {best_accuracy_train}")
    print("Confusion Matrix for the best model on test set:\n", best_confusion_matrix_test)
    print("Confusion Matrix for the best model on training set:\n", best_confusion_matrix_train)
    print(f"Test Precision: {precision_test}, Test Recall: {recall_test}, Test F1: {f1_test}, Test Accuracy: {accuracy_test}")
    print(f"Training Precision: {precision_train}, Training Recall: {recall_train}, Training F1: {f1_train}, Training Accuracy: {accuracy_train}")

    if len(X_pool) > 0 and len(y_pool) > 0:
        query_indices = np.random.choice(X_pool.index, min(len(X_pool), n_queries_per_iteration), replace=False)
        X_queried = X_pool.loc[query_indices]
        y_queried = y_pool.loc[query_indices]

        X_pool = X_pool.drop(query_indices)
        y_pool = y_pool.drop(query_indices)
    else:
        print("No more samples in pool to query.")
        break

    print(f"End of iteration {iteration + 1}.")

print("End of Active Learning.")


=====

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming features_df and target_binary are defined elsewhere in your notebook
X_train, X_pool, y_train, y_pool = train_test_split(features_df, target_binary, test_size=0.5, random_state=42)
X_pool, X_test, y_pool, y_test = train_test_split(X_pool, y_pool, test_size=0.2, random_state=42)

best_accuracy = 0
best_params = {}

max_depths = [2]
learning_rates = [0.08]
n_estimators_list = [30]

n_active_learning_iterations = 12
n_queries_per_iteration = 12

for iteration in range(n_active_learning_iterations):
    print(f"Active learning iteration: {iteration + 1}")

    for max_depth in max_depths:
        for learning_rate in learning_rates:
            for n_estimators in n_estimators_list:
                xgb_classifier = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1,
                    reg_alpha=0,
                    objective='binary:logistic',
                    random_state=42
                )

                xgb_classifier.fit(X_train, y_train)
                y_pred = xgb_classifier.predict(X_test)

                # Calculate and print accuracy, precision, recall, and F1 score
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary')
                recall = recall_score(y_test, y_pred, average='binary')
                f1 = f1_score(y_test, y_pred, average='binary')

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'n_estimators': n_estimators,
                    }
                    print(f"New best parameters: {best_params} with accuracy: {best_accuracy}, precision: {precision}, recall: {recall}, F1: {f1}")

    # Check if there are enough samples in the pool to perform the next query
    n_samples_left = len(X_pool)
    if n_samples_left >= n_queries_per_iteration:
        query_size = n_queries_per_iteration
    else:
        query_size = n_samples_left  # If fewer than n_queries_per_iteration, use all the remaining samples

    # If there are samples left to query, perform the active learning step
    if query_size > 0:
        indices = np.random.choice(range(n_samples_left), query_size, replace=False)
        X_queried = X_pool.iloc[indices]
        y_queried = y_pool.iloc[indices]

        X_pool = X_pool.drop(X_pool.index[indices])
        y_pool = y_pool.drop(y_pool.index[indices])

        X_train = pd.concat([X_train, X_queried])
        y_train = pd.concat([y_train, y_queried])
    else:
        print("No more samples in pool to query.")
        break

    print(f"End of iteration {iteration + 1}. Best accuracy so far: {best_accuracy}")

print(f"Final best parameters: {best_params} with accuracy: {best_accuracy}")


=====

# Re-train the model with the best parameters on the training set
best_model = xgb.XGBClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    reg_alpha=0,
    objective='binary:logistic',
    random_state=42
)

best_model.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = best_model.predict(X_train)

# Make predictions on the testing set
y_test_pred = best_model.predict(X_test)

# Calculate confusion matrices
from sklearn.metrics import confusion_matrix
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)

# Display the results
print("Training Set Confusion Matrix:\n", confusion_matrix_train)
print("Testing Set Confusion Matrix:\n", confusion_matrix_test)

# Extract and display true/false positives/negatives from the confusion matrix
tn_train, fp_train, fn_train, tp_train = confusion_matrix_train.ravel()
tn_test, fp_test, fn_test, tp_test = confusion_matrix_test.ravel()

print(f"Training Set - True Positives: {tp_train}, False Positives: {fp_train}, True Negatives: {tn_train}, False Negatives: {fn_train}")
print(f"Testing Set - True Positives: {tp_test}, False Positives: {fp_test}, True Negatives: {tn_test}, False Negatives: {fn_test}")


=====

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(features_df, target_binary)

scaler = RobustScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

X_train, X_test, y_train, y_test = train_test_split(X_resampled_scaled, y_resampled, test_size=0.2, random_state=42)

max_depths = [4, 5, 6, 7, 8, 9, 10]
learning_rates = [0.0001, 0.005, 0.001, 0.09, 0.08, 0.075, 0.07, 0.065, 0.06, 0.055, 0.05, 0.04, 0.03, 0.02, 0.01, 0.1, 0.2, 0.5]
n_estimators_list = [60, 100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500]

best_accuracy = 0
best_params = {}

for max_depth in max_depths:
    for learning_rate in learning_rates:
        for n_estimators in n_estimators_list:
            xgb_classifier = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1,
                reg_alpha=0,
                objective='binary:logistic',
                random_state=42
            )
            xgb_classifier.fit(X_train, y_train)

            y_pred = xgb_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'n_estimators': n_estimators,
                }
                print(f"New best parameters: {best_params} with accuracy: {best_accuracy}")

print(f"Best parameters: {best_params} with accuracy: {best_accuracy}")

=====

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('featurized_data.csv')

# Assuming the last column is the target variable
X = data.iloc[:, :-1]  # Feature matrix
y = data.iloc[:, -1]  # Target variable

results = []  # To store results of each iteration

# Set a seed for reproducibility
np.random.seed(42)

for i in range(11):
    # Generate a random state
    random_state = np.random.randint(0, 1000)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Initialize a model (using RandomForest for illustration)
    model = RandomForestClassifier(n_estimators=100)

    # Setup GridSearchCV to tune hyperparameters and use 10-fold CV
    param_grid = {
        'max_depth': [10, 20, 30, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Evaluate the model
    best_model = grid_search.best_estimator_
    predictions_train = best_model.predict(X_train)
    predictions_test = best_model.predict(X_test)

    # Calculate metrics
    accuracy_train = accuracy_score(y_train, predictions_train)
    precision_train = precision_score(y_train, predictions_train, average='weighted')
    recall_train = recall_score(y_train, predictions_train, average='weighted')
    f1_score_train = f1_score(y_train, predictions_train, average='weighted')
    confusion_matrix_train = confusion_matrix(y_train, predictions_train)

    accuracy_test = accuracy_score(y_test, predictions_test)
    precision_test = precision_score(y_test, predictions_test, average='weighted')
    recall_test = recall_score(y_test, predictions_test, average='weighted')
    f1_score_test = f1_score(y_test, predictions_test, average='weighted')
    confusion_matrix_test = confusion_matrix(y_test, predictions_test)

    results.append({
        'random_state': random_state,
        'best_params': grid_search.best_params_,
        'accuracy_train': accuracy_train,
        'precision_train': precision_train,
        'recall_train': recall_train,
        'f1_score_train': f1_score_train,
        'confusion_matrix_train': confusion_matrix_train,
        'accuracy_test': accuracy_test,
        'precision_test': precision_test,
        'recall_test': recall_test,
        'f1_score_test': f1_score_test,
        'confusion_matrix_test': confusion_matrix_test
    })

# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame(results)

# Print the results
print(results_df)


=====

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# List of specified random states
random_states = [0, 12, 22, 42, 100, 124, 300, 590, 610, 700, 920]

# List of data files
data_files = [
    'featurized_data_8.csv',
    'featurized_data_16.csv',
    'featurized_data_32.csv',
    'featurized_data_64.csv',
    'featurized_data_128.csv',
    'featurized_data_256.csv',
    'featurized_data_512.csv',
    'featurized_data_1024.csv'
]

def calculate_metrics_from_confusion_matrix(cm):
    # For binary classification
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    else:
        # For multi-class classification
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1_score = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1_score

for file in data_files:
    # Load the dataset
    data = pd.read_csv(file)

    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]  # Target variable

    results = []  # To store results of each iteration

    for random_state in random_states:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        # Initialize a model (using XGBoost for illustration)
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions_train = model.predict(X_train)
        predictions_test = model.predict(X_test)

        # Compute confusion matrices
        confusion_matrix_train = confusion_matrix(y_train, predictions_train)
        confusion_matrix_test = confusion_matrix(y_test, predictions_test)

        # Calculate metrics from confusion matrices
        y_true = y_test  # Set y_true for multi-class calculation
        y_pred = predictions_test  # Set y_pred for multi-class calculation
        accuracy_train, precision_train, recall_train, f1_score_train = calculate_metrics_from_confusion_matrix(confusion_matrix_train)
        accuracy_test, precision_test, recall_test, f1_score_test = calculate_metrics_from_confusion_matrix(confusion_matrix_test)

        results.append({
            'random_state': random_state,
            'accuracy_train': accuracy_train,
            'precision_train': precision_train,
            'recall_train': recall_train,
            'f1_score_train': f1_score_train,
            'confusion_matrix_train': confusion_matrix_train,
            'accuracy_test': accuracy_test,
            'precision_test': precision_test,
            'recall_test': recall_test,
            'f1_score_test': f1_score_test,
            'confusion_matrix_test': confusion_matrix_test
        })

    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)

    # Print the results
    print(f'Results for file: {file}')
    print(results_df)

    # Plot confusion matrices as heatmaps
    for i, result in enumerate(results):
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        sns.heatmap(result['confusion_matrix_train'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Train Confusion Matrix - Iteration {i+1} ({file})')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')

        plt.subplot(1, 2, 2)
        sns.heatmap(result['confusion_matrix_test'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Test Confusion Matrix - Iteration {i+1} ({file})')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')

        plt.tight_layout()
        plt.show()
