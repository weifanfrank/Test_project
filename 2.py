pip install sklearn

=====

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    else:
        return np.array([0]*n_bits)


file_path = '/Drugs_with_isomeric_smile.xlsx'
data = pd.read_excel(file_path)


scaler = MinMaxScaler()
numeric_features = ['1st Replicate PrecA-gfp (GFP)', '2nd Replicate PrecA-gfp (GFP)']
data[numeric_features] = scaler.fit_transform(data[numeric_features])
print(data[numeric_features])


data['MorganFP'] = data['Isomeric smiles '].apply(smiles_to_morgan)
fp_df = pd.DataFrame(data['MorganFP'].tolist())
fp_df.columns = [f'Bit_{i}' for i in range(fp_df.shape[1])]


data_with_fp = pd.concat([data, fp_df], axis=1).drop(columns=['MorganFP'])


output_path = '/content/sample_data/Drugs_with_features.xlsx'
data_with_fp.to_excel(output_path, index=False)


print(f'Data with features saved to {output_path}')


=====

Use models like random forest to get feature important scores. Based on scores decide which features to keep

-----

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

file_path = '/content/sample_data/Drugs_with_filtered_features.xlsx'
data_filtered = pd.read_excel(file_path)

target_variables = ['1st Replicate PrecA-gfp (GFP)', '2nd Replicate PrecA-gfp (GFP)']

X_numeric = data_filtered[remaining_features]
y = data_filtered[target_variables]

rf_model = RandomForestRegressor()

rf_model.fit(X_numeric, y)

feature_importance = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': remaining_features, 'Importance': feature_importance})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

top_features = feature_importance_df.head(100)['Feature']

print("Remaining Features to Keep:")
print(top_features)

original_file_path = '/content/sample_data/Drugs_with_features.xlsx'
original_data = pd.read_excel(original_file_path)

non_numeric_columns = ["Drug Name", "Drug Number", "Isomeric smiles "]

data_selected_features = original_data[non_numeric_columns].join(data_filtered[['1st Replicate PrecA-gfp (GFP)', '2nd Replicate PrecA-gfp (GFP)'] + top_features.tolist()])

data_selected_features = data_selected_features[['Drug Name', 'Drug Number', '1st Replicate PrecA-gfp (GFP)', '2nd Replicate PrecA-gfp (GFP)', 'Isomeric smiles '] + top_features.tolist()]

output_path = '/content/sample_data/Drugs_with_selected_features.xlsx'
data_selected_features.to_excel(output_path, index=False)

print(f'Data with selected features saved to {output_path}')

=====

Rank features by recursively considering smaller and smaller sets of features

-----

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

# Load the dataset with selected features
file_path = '/content/sample_data/Drugs_with_selected_features.xlsx'
data_selected_features = pd.read_excel(file_path)

# Define the target variables
target_variables = ['1st Replicate PrecA-gfp (GFP)', '2nd Replicate PrecA-gfp (GFP)']

# Get the features to rank (excluding the target variables and non-numeric columns)
features_to_rank = [col for col in data_selected_features.columns if col not in target_variables and col not in ['Drug Name', 'Drug Number', 'Isomeric smiles ']]

# Prepare the input features (X) and target (y)
X = data_selected_features[features_to_rank]
y = data_selected_features[target_variables]

# Create a Random Forest model for feature ranking
rf_model = RandomForestRegressor()

# Create an RFE selector with the Random Forest model
rfe = RFE(rf_model, step=1)

# Fit the RFE selector to the data
rfe.fit(X, y)

# Get the ranking of features
feature_ranking = rfe.ranking_

# Create a DataFrame to store feature names and their ranks
feature_ranking_df = pd.DataFrame({'Feature': features_to_rank, 'Rank': feature_ranking})

# Sort the features by rank in ascending order
feature_ranking_df = feature_ranking_df.sort_values(by='Rank', ascending=True)

# Save the ranked features to a new Excel file
output_path = '/content/sample_data/Drugs_with_ranked_features.xlsx'
feature_ranking_df.to_excel(output_path, index=False)

print(f'Ranked features saved to {output_path}')

=====

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np

df = pd.read_excel('/content/Drugs_with_isomeric_smile.xlsx')


non_numeric_columns = df.select_dtypes(include=['object']).columns


if len(non_numeric_columns) > 0:
    df = pd.get_dummies(df, columns=non_numeric_columns)

#print(df.head())
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print(np.shape(X))
#print('x',X)
#print('y',y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


logreg_model = LogisticRegression()
logreg_model.fit(X_train_scaled, y_train)


y_pred_lr = logreg_model.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
report_lr = classification_report(y_test, y_pred_lr)


svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)


y_pred_svm = svm_model.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)


performance = {
    'Logistic Regression': {
        'Accuracy': accuracy_lr,
        'Classification Report': report_lr
    },
    'Support Vector Machine': {
        'Accuracy': accuracy_svm,
        'Classification Report': report_svm
    }
}

=====

import pandas as pd
from sklearn.decomposition import PCA


df = pd.read_excel('/content/Cleaned_Drugs_with_Averages.xlsx')


bit_columns = [col for col in df.columns if 'Bit_' in col]
X = df[bit_columns]


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

pd.set_option('display.max_rows', 360)

print(df[['Drug Name', 'PCA1', 'PCA2']])

=====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

# Load your data
df = pd.read_excel('/content/Cleaned_Drugs_with_Averages.xlsx')

# Preprocess and setup PCA
bit_columns = [col for col in df.columns if 'Bit_' in col]
X = df[bit_columns]
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Add PCA results to the dataframe
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Setup binary target variable
y = df['average']
median_gfp = y.median()
y_binary = (y > median_gfp).astype(int)

# Select features and split data
X_pca_features = df[['PCA1', 'PCA2']]
X_train, X_test, y_train_binary, y_test_binary = train_test_split(X_pca_features, y_binary, test_size=0.2, random_state=42)

# Define a function to perform Grid Search
def perform_grid_search(X_train, y_train):
    # Define the parameter grid to search
    param_grid = {
        'C': np.logspace(-4, 4, 20),
        'penalty': ['l1', 'l2'],  # 'elasticnet' and 'none' require specific solvers
        'solver': ['liblinear', 'saga'],  # 'liblinear' works well with small datasets and 'saga' is a good choice for 'l1' and 'elasticnet'
        'max_iter': [1000, 2000],
        'class_weight': [None, 'balanced'],  # For handling imbalanced data
    }

    # Perform Grid Search with 5-fold cross-validation
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_, grid_search.best_score_

# Perform the grid search
best_params, best_score = perform_grid_search(X_train, y_train_binary)

# Print the results
print(f"Best parameters found: {best_params}")
print(f"Best cross-validation score: {best_score}")

# Train the best model
best_log_reg = LogisticRegression(
    C=best_params['C'],
    penalty=best_params['penalty'],
    solver=best_params['solver'],
    max_iter=best_params['max_iter'],
    class_weight=best_params['class_weight']
)
best_log_reg.fit(X_train, y_train_binary)

# Predict probabilities
y_pred_probabilities = best_log_reg.predict_proba(X_test)

# Set a custom threshold (e.g., 0.4)
custom_threshold = 0.5

# Classify based on the custom threshold
y_pred_binary_custom = (y_pred_probabilities[:, 1] >= custom_threshold).astype(int)

# Print the evaluation metrics and confusion matrix
print(classification_report(y_test_binary, y_pred_binary_custom))
print("Mean Squared Error:", mean_squared_error(y_test_binary, y_pred_binary_custom))
print("Training Score:", best_log_reg.score(X_train, y_train_binary))
print("Test Score:", best_log_reg.score(X_test, y_test_binary))
print("Confusion Matrix:")
print(confusion_matrix(y_test_binary, y_pred_binary_custom))

# Plotting Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    best_log_reg, X_pca_features, y_binary, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.01, 1.0, 50))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

plt.plot(train_sizes, train_mean, label="Training score")
plt.plot(train_sizes, test_mean, label="Cross-validation score")

plt.title("Learning Curve")
plt.xlabel("Training Size")
plt.ylabel("Score")
plt.legend(loc="best")
plt.show()

=====

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, mean_squared_error, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Set a global random seed for reproducibility
np.random.seed(42)

# Load your data
df = pd.read_excel('/content/Cleaned_Drugs_with_Averages.xlsx')

bit_columns = [col for col in df.columns if 'Bit_' in col]
X = df[bit_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# If PCA has a randomized component, set the random_state here too
pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_scaled)

y = df['average']

median_gfp = y.median()
y_binary = (y > median_gfp).astype(int)

# The random_state here ensures the same split every time
X_train, X_test, y_train_binary, y_test_binary = train_test_split(X_pca, y_binary, test_size=0.2, random_state=42)

# Create an SVC classifier
svc = SVC()

# Define hyperparameter search space
parameters = {
    'C': [0.5, 5, 50],  # Regularization parameter
    'kernel': ['poly', 'rbf'],  # Type of kernel
    'gamma': ['scale', 'auto'],  # Kernel coefficient
}

# Perform grid search
grid_search = GridSearchCV(svc, parameters, cv=3, scoring='f1_macro')
grid_search.fit(X_train, y_train_binary)

# Get the best parameters and estimator
best_params = grid_search.best_params_
best_svc = grid_search.best_estimator_

# Predict on test data
y_pred_best_svc = best_svc.predict(X_test)

# Calculate and print the classification report
report_best_svc = classification_report(y_test_binary, y_pred_best_svc, output_dict=True)
report_best_svc_df = pd.DataFrame(report_best_svc).transpose()
print(report_best_svc_df[['precision', 'recall', 'f1-score']])

# Calculate and print the confusion matrix
confusion = confusion_matrix(y_test_binary, y_pred_best_svc)
print("Confusion Matrix:")
print(confusion)

# Calculate and print Mean Squared Error
mse = mean_squared_error(y_test_binary, y_pred_best_svc)
print("Mean Squared Error:", mse)

# Create a learning curve
train_sizes, train_scores, test_scores = learning_curve(best_svc, X_train, y_train_binary, cv=3, scoring='f1_macro')
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

# Print training and test scores
print("Training F1 Score:", train_scores_mean[-1])
print("Validation F1 Score:", test_scores_mean[-1])

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.title("Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()

plt.plot(train_sizes, train_scores_mean, label="Training F1 Score", color="r")
plt.plot(train_sizes, test_scores_mean, label="Validation F1 Score", color="g")

plt.legend(loc="best")
plt.show()

=====

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve

# Set the global random state
random_state = 42
np.random.seed(random_state)

# Load your data
df = pd.read_excel('/content/Cleaned_Drugs_with_Averages.xlsx')

# Select features and target variable
bit_columns = [col for col in df.columns if 'Bit_' in col]
X = df[bit_columns]
y = df['average']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Binarize target variable
median_gfp = y.median()
y_binary = (y > median_gfp).astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train_binary, y_test_binary = train_test_split(X_pca, y_binary, test_size=0.2, random_state=random_state)

# Gaussian Process Classifier with RBF kernel
kernel = 0.01 * RBF(length_scale=2.0)
gp_classifier = GaussianProcessClassifier(kernel=kernel, random_state=random_state)

# Fit the classifier to the training data
gp_classifier.fit(X_train, y_train_binary)

# Predict on the test set
y_pred_gp = gp_classifier.predict(X_test)

# Classification report
report_gp = classification_report(y_test_binary, y_pred_gp, output_dict=True)
report_gp_df = pd.DataFrame(report_gp).transpose()

# Mean Squared Error
mse_gp = mean_squared_error(y_test_binary, y_pred_gp)

# Confusion matrix
conf_matrix = confusion_matrix(y_test_binary, y_pred_gp)

# Print results
print(report_gp_df[['precision', 'recall', 'f1-score']])
print("Mean Squared Error (Gaussian Process):", mse_gp)
print("Confusion Matrix:")
print(conf_matrix)

# Learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

title = "Learning Curve (Gaussian Process Classifier)"
cv = 5  # You can specify the number of cross-validation folds here if needed
plot_learning_curve(gp_classifier, title, X_train, y_train_binary, cv=cv)

# Print training and validation scores
train_score = gp_classifier.score(X_train, y_train_binary)
test_score = gp_classifier.score(X_test, y_test_binary)
print("Training Score:", train_score)
print("Validation Score:", test_score)

plt.show()
