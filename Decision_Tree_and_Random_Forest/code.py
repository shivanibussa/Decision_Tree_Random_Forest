# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree

# Load the dataset

df = pd.read_csv("/content/dataset.csv")
print(len(df))
# Pre-processing the dataset
df.replace('unknown', np.nan, inplace=True)
# Check for null values in each column
null_values_per_column = df.isnull().sum()

# Filter columns with null values
columns_with_null_values = null_values_per_column[null_values_per_column > 0]

print("Columns with null values:")
print(columns_with_null_values)

# Impute missing values for job and education with mode

# Since there are relatively few null values in this column (112),
# we can consider imputing the missing values with the most frequent job category (mode).
df['job'].fillna(df['job'].mode()[0], inplace=True)

# With 1073 null values, we can use mode imputation to fill in the missing values with the most frequent education level.
df['education'].fillna(df['education'].mode()[0], inplace=True)

# Tried considering poutcome got an accuracy of 0.7579 and 0.7583 So dropped it as results are same.
# df['poutcome'].fillna(df['poutcome'].mode()[0], inplace = True)

# Drop contact and poutcome columns

# There are 4313 null values in contact column, which is a significant portion of the data so dropping it.
# With 13259 null values, in poutcome column has a large number of missing values.
# We consider dropping this column as it's not essential for the analysis or model,
df.drop(columns=['contact', 'poutcome'], inplace=True)
# df.drop(columns=['contact'], inplace=True)

# Impute missing values for day and month with mean and mode respectively
# With 306 null values in day column, we used mean imputation to fill in the missing values, depending on the distribution of the data as missing is random.
# Similarly, with 314 null values in month column, we can use mode imputation to fill in the missing values with the most frequent month.
df['day'].fillna(df['day'].mean(), inplace=True)
df['month'].fillna(df['month'].mode()[0], inplace=True)

# Encoding categorical columns into numerical format using label encoding, allowing the data to be used in machine learning models that require numerical input.
encoder = LabelEncoder()
df['job'] = encoder.fit_transform(df['job'])
df['marital'] = encoder.fit_transform(df['marital'])
df['education'] = encoder.fit_transform(df['education'])
df['default'] = encoder.fit_transform(df['default'])
df['housing'] = encoder.fit_transform(df['housing'])
df['loan'] = encoder.fit_transform(df['loan'])
# df['poutcome'] = encoder.fit_transform(df['poutcome'])
print(len(df))
from sklearn.ensemble import RandomForestClassifier

# One-hot encoding for categorical variables
X = pd.get_dummies(df.drop(columns=['y']), drop_first=True)
y = df['y']

# Initialize Random Forest classifier

# The Random Forest classifier identifies the six most influential attributes by calculating feature importances during training.
# These importances represent each feature's contribution to predicting the target variable.
# The top six features are selected based on their importance scores, providing insights into the key factors driving the prediction outcome.

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fitting the model
rf_classifier.fit(X, y)

# Getting feature importances of various features
feature_importances = rf_classifier.feature_importances_

# Creating a dataframe to store feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sorting features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Printing the top six most influential features
top_six_features = feature_importance_df.head(6)
print(top_six_features)

# Visualization 1: Age distribution with respect to target variable
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='y', multiple='stack', bins=20)
plt.title('Age Distribution with Respect to Subscription')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Subscription', labels=['No', 'Yes'])
plt.show()
# Explanation: This histogram shows the distribution of ages for clients who subscribed (Yes) and those who didn't (No).
# It helps to understand if there's any age group more likely to subscribe to the term deposit.

# Visualization 2: Education level distribution with respect to target variable
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='education', hue='y')
plt.title('Education Level Distribution with Respect to Subscription')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.legend(title='Subscription', labels=['No', 'Yes'])
plt.xticks(ticks=np.arange(4), labels=['Primary', 'Secondary', 'Tertiary', 'Unknown'])
plt.show()
# Explanation: This countplot displays the distribution of education levels for clients who subscribed (Yes) and those who didn't (No).
# It helps to identify if there's any correlation between education level and subscribing to the term deposit.

# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=['month'])

# Splitting dataset into training and testing sets with the encoded dataframe as 75% training and 25% as testing set.
X_encoded = df_encoded.drop(columns=['y'])  # Features
y_encoded = df_encoded['y']  # Target variable
X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(X_encoded, y_encoded, test_size=0.25, random_state=42)

# Initializing the classifiers with different parameters
dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=5)
dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=5)

# Fitting the classifiers
dt_gini.fit(X_train_encoded, y_train_encoded)
dt_entropy.fit(X_train_encoded, y_train_encoded)

# Predictions with gini index and entropy
y_pred_gini = dt_gini.predict(X_test_encoded)
y_pred_entropy = dt_entropy.predict(X_test_encoded)

# Confusion matrix and classification report for gini criterion
print("Confusion Matrix (Gini):")
print(confusion_matrix(y_test_encoded, y_pred_gini))
# graphics of the confusion matrix
plt.figure(figsize=(2, 2))
sns.heatmap(confusion_matrix(y_test_encoded, y_pred_entropy), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
print("Classification Report (Gini):")
print(classification_report(y_test_encoded, y_pred_gini))

# Confusion matrix and classification report for entropy criterion
print("Confusion Matrix (Entropy):")
print(confusion_matrix(y_test_encoded, y_pred_entropy))
# graphics of the confusion matrix
plt.figure(figsize=(2, 2))
sns.heatmap(confusion_matrix(y_test_encoded, y_pred_entropy), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
print("Classification Report (Entropy):")
print(classification_report(y_test_encoded, y_pred_entropy))

# Comparing the results of gini vs entropy
print("Accuracy (Gini):", dt_gini.score(X_test_encoded, y_test_encoded))
print("Accuracy (Entropy):", dt_entropy.score(X_test_encoded, y_test_encoded))

# Decision Tree Visualization with depth of 5
plt.figure(figsize=(100, 100))
plot_tree(dt_gini, filled=True, feature_names=X_encoded.columns, class_names=['No', 'Yes'], max_depth=5)
plt.title("Decision Tree Visualization (Gini)")
plt.show()

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train_encoded, y_train_encoded)
rfc_pred = rfc.predict(X_test_encoded)

# Evaluating the Decision Tree Model and Random Forest Classifier
print("Decision Tree Accuracy:", dt_gini.score(X_test_encoded, y_test_encoded))
print("Random Forest Accuracy:", rfc.score(X_test_encoded, y_test_encoded))

# ROC Curves for both classifiers
# Encoding target variable into numeric values
label_encoder = LabelEncoder()
y_test_encoded_numeric = label_encoder.fit_transform(y_test_encoded)

# Decision Tree
dt_probs = dt_gini.predict_proba(X_test_encoded)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test_encoded_numeric, dt_probs)

# Random Forest
rf_probs = rfc.predict_proba(X_test_encoded)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test_encoded_numeric, rf_probs)

# Plotting ROC curves
plt.figure(figsize=(10, 8))
plt.plot(fpr_dt, tpr_dt, linestyle='--', label='Decision Tree')
plt.plot(fpr_rf, tpr_rf, marker='.', label='Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()