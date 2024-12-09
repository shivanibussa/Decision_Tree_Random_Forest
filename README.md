# Decision Tree and Random Forest Classification

This project demonstrates the implementation of Decision Tree and Random Forest classifiers using a dataset to predict the likelihood of a target event (`y`). The steps involve data preprocessing, feature selection, model training, evaluation, and visualization.

## Project Workflow

### 1. **Dataset Loading**
- The dataset is loaded and its size is checked.
- Missing values are identified in columns, and preprocessing steps are applied to handle them.

### 2. **Data Preprocessing**
- Missing values in categorical columns (`job`, `education`) are imputed with the mode.
- Columns with excessive missing values (`contact`, `poutcome`) are dropped.
- Missing values in numerical columns (`day`, `month`) are imputed with their respective mean or mode.
- Categorical columns are encoded using `LabelEncoder`.

### 3. **Feature Selection**
- The importance of each feature is calculated using a Random Forest classifier.
- The top six features are identified and visualized for analysis.

### 4. **Data Visualization**
- Age distribution is plotted with respect to the target variable (`y`), showing how age groups correlate with subscription likelihood.
- Education levels are visualized to understand their relationship with the target variable.

### 5. **Model Training and Evaluation**
- The dataset is split into training (75%) and testing (25%) sets.
- Two classifiers, Decision Tree (with Gini index and Entropy) and Random Forest, are trained on the data.
- The following evaluations are performed:
  - Confusion Matrix
  - Classification Report
  - Accuracy Score
  - ROC Curves

### 6. **Decision Tree Visualization**
- The Decision Tree (depth = 5) is visualized to illustrate the decision-making process.

### 7. **Comparison of Classifiers**
- The Decision Tree and Random Forest classifiers are compared based on their accuracy and ROC curves.

## Installation and Usage

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
