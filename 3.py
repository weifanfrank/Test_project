import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, confusion_matrix
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, models

df = pd.read_excel('/content/Cleaned_Drugs_with_Averages.xlsx')

bit_columns = [col for col in df.columns if 'Bit_' in col]
X = df[bit_columns]
y = df['average']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_scaled)

median_gfp = y.median()
y_binary = (y > median_gfp).astype(int)

X_train, X_test, y_train_binary, y_test_binary = train_test_split(X_pca, y_binary, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train_binary, epochs=45, batch_size=32, validation_split=0.2)

y_pred_prob = model.predict(X_test)
y_pred_binary = (y_pred_prob > 0.5).astype(int)

report_dl = classification_report(y_test_binary, y_pred_binary, output_dict=True)
report_dl_df = pd.DataFrame(report_dl).transpose()
print(report_dl_df[['precision', 'recall', 'f1-score']])

mse_dl = mean_squared_error(y_test_binary, y_pred_binary)
print("Mean Squared Error (Deep Learning):", mse_dl)

# Calculate and print the confusion matrix
confusion_mat = confusion_matrix(y_test_binary, y_pred_binary)
print("Confusion Matrix:")
print(confusion_mat)

# Print training and validation scores
train_loss, train_acc = model.evaluate(X_train, y_train_binary, verbose=0)
val_loss, val_acc = model.evaluate(X_test, y_test_binary, verbose=0)

print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

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
pca = PCA(n_components=10, random_state=42)  # Set random_state if PCA uses randomness
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
rf_grid_search = GridSearchCV(rf_classifier, rf_parameters, cv=5, scoring='f1_macro')
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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, confusion_matrix
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from sklearn.utils import check_random_state
random_state = check_random_state(42)

df = pd.read_excel('/content/Cleaned_Drugs_with_Averages.xlsx')

bit_columns = [col for col in df.columns if 'Bit_' in col]
X = df[bit_columns]
y = df['average']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=12, random_state=42)
X_pca = pca.fit_transform(X_scaled)

median_gfp = y.median()
y_binary = (y > median_gfp).astype(int)

X_train, X_test, y_train_binary, y_test_binary = train_test_split(X_pca, y_binary, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train_binary, epochs=45, batch_size=32, validation_split=0.2)

y_pred_prob = model.predict(X_test)
y_pred_binary = (y_pred_prob > 0.5).astype(int)

report_dl = classification_report(y_test_binary, y_pred_binary, output_dict=True)
report_dl_df = pd.DataFrame(report_dl).transpose()
print(report_dl_df[['precision', 'recall', 'f1-score']])

mse_dl = mean_squared_error(y_test_binary, y_pred_binary)
print("Mean Squared Error (Deep Learning):", mse_dl)

# Calculate and print the confusion matrix
confusion_mat = confusion_matrix(y_test_binary, y_pred_binary)
print("Confusion Matrix:")
print(confusion_mat)

# Print training and validation scores
train_loss, train_acc = model.evaluate(X_train, y_train_binary, verbose=0)
val_loss, val_acc = model.evaluate(X_test, y_test_binary, verbose=0)

print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

=====

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, confusion_matrix

import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from sklearn.utils import check_random_state
random_state = check_random_state(42)

df = pd.read_excel('/content/Cleaned_Drugs_with_Averages.xlsx')

bit_columns = [col for col in df.columns if 'Bit_' in col]
X = df[bit_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=12, random_state=42)
X_pca = pca.fit_transform(X_scaled)

y = df['average']

median_gfp = y.median()
y_binary = (y > median_gfp).astype(int)

X_train, X_test, y_train_binary, y_test_binary = train_test_split(X_pca, y_binary, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier()

rf_parameters = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid_search = GridSearchCV(rf_classifier, rf_parameters, cv=5, scoring='f1_macro')
rf_grid_search.fit(X_train, y_train_binary)

best_rf_params = rf_grid_search.best_params_
best_rf_classifier = rf_grid_search.best_estimator_

y_pred_best_rf = best_rf_classifier.predict(X_test)

report_best_rf = classification_report(y_test_binary, y_pred_best_rf, output_dict=True)
report_best_rf_df = pd.DataFrame(report_best_rf).transpose()
print(report_best_rf_df[['precision', 'recall', 'f1-score']])

mse_rf = mean_squared_error(y_test_binary, y_pred_best_rf)
print("Mean Squared Error (Random Forest):", mse_rf)

# Calculate and print the confusion matrix
confusion_mat = confusion_matrix(y_test_binary, y_pred_best_rf)
print("Confusion Matrix:")
print(confusion_mat)

# Print validation and test scores
validation_score = rf_grid_search.best_score_
test_score = best_rf_classifier.score(X_test, y_test_binary)
print(f"Validation Score (F1 macro): {validation_score:.4f}")
print(f"Test Score (Accuracy): {test_score:.4f}")

=====

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, confusion_matrix
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

df = pd.read_excel('/content/Cleaned_Drugs_with_Averages.xlsx')

bit_columns = [col for col in df.columns if 'Bit_' in col]
X = df[bit_columns]
y = df['average']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

median_gfp = y.median()
y_binary = (y > median_gfp).astype(int)

X_train, X_test, y_train_binary, y_test_binary = train_test_split(X_pca, y_binary, test_size=0.2, random_state=42)

# Define hyperparameters
nodes_in_layers = [128, 64, 32]
learning_rates = [0.01, 0.001, 0.0001]
dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
batch_sizes = [32, 64, 128]
epoch_values = [30, 35, 40, 45, 50, 55, 60, 65, 70, 80]

# Iterate over hyperparameter combinations
for nodes in nodes_in_layers:
    for lr in learning_rates:
        for dropout_rate in dropout_rates:
            for batch_size in batch_sizes:
                for epochs in epoch_values:
                    # Create the model
                    model = models.Sequential([
                        layers.Dense(nodes, activation='relu', input_shape=(X_train.shape[1],)),
                        layers.Dropout(dropout_rate),
                        layers.Dense(nodes, activation='sigmoid'),
                        layers.Dropout(dropout_rate),
                        layers.Dense(nodes, activation='relu'),
                        layers.Dropout(dropout_rate),
                        layers.Dense(1, activation='sigmoid')
                    ])

                    # Compile the model
                    model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])

                    # Train the model
                    history = model.fit(X_train, y_train_binary, epochs=epochs, batch_size=batch_size, validation_split=0.2)

                    # Make predictions
                    y_pred_prob = model.predict(X_test)
                    y_pred_binary = (y_pred_prob > 0.5).astype(int)

                    # Evaluate the model
                    report_dl = classification_report(y_test_binary, y_pred_binary, output_dict=True)
                    report_dl_df = pd.DataFrame(report_dl).transpose()
                    print(f"Nodes: {nodes}, LR: {lr}, Dropout: {dropout_rate}, Batch Size: {batch_size}, Epochs: {epochs}")
                    print(report_dl_df[['precision', 'recall', 'f1-score']])

                    mse_dl = mean_squared_error(y_test_binary, y_pred_binary)
                    print("Mean Squared Error (Deep Learning):", mse_dl)

                    # Calculate and print the confusion matrix
                    confusion_mat = confusion_matrix(y_test_binary, y_pred_binary)
                    print("Confusion Matrix:")
                    print(confusion_mat)

                    # Print training and validation scores
                    train_loss, train_acc = model.evaluate(X_train, y_train_binary, verbose=0)
                    val_loss, val_acc = model.evaluate(X_test, y_test_binary, verbose=0)

                    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
                    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
                    print("-------------------------------------------------------------")
