# 🩺 Diabetes Prediction using KNN — ML Project 4

A K-Nearest Neighbors classification model to predict
diabetes risk based on medical measurements.

## Dataset
[Pima Indians Diabetes Dataset — Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
768 rows | 8 features | Binary target (0=No Diabetes, 1=Diabetes)

## Problem Type
**Binary Classification** using `KNeighborsClassifier`

## Steps Followed

| Step | Task |
|---|---|
| EDA | Shape, dtypes, missing values, distributions |
| Visualisation | Target distribution, correlation heatmap, feature histograms |
| Preprocessing | Replace zero values with NaN → median imputation |
| Scaling | StandardScaler (critical for KNN) |
| Model | KNeighborsClassifier with k=3,5,7,9,11,15,21 |
| Evaluation | Accuracy, Confusion Matrix, ROC-AUC, Classification Report |
| Cross Validation | 5-fold CV on best k |

## Results

| k | Accuracy | ROC-AUC |
|---|---|---|
| 3 | 72.08% | 0.7675 |
| 5 | 75.32% | 0.7899 |
| 7 | 72.73% | 0.7900 |
| 9 | 74.03% | 0.7951 |
| 11 | 73.38% | 0.7880 |
| 15 | 74.03% | 0.8068 |
| **21** | **75.97%** | **0.8120** ✅|

**Best k = 21 → Accuracy: 75.97% | ROC-AUC: 0.8120**

## Key Insights
- Glucose and BMI are strongest predictors of diabetes
- Zero values in medical columns = missing data, not real zeros
- Higher k gave better performance — smoother decision boundary reduced overfitting

## Tech Stack
`Python` `scikit-learn` `pandas` `NumPy` `matplotlib` `seaborn`
