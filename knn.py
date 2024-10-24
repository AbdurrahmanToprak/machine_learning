import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from logistic_regression import cv_results

#######################
# EDA
#######################
pd.set_option('display.max_columns', None)

df = pd.read_csv('datasets/diabetes.csv')

df.shape
df.head()
df.describe().T
df["Outcome"].value_counts()

#######################
# Data Preprocessing & Feature Engineering
#######################

y = df["Outcome"]

X = df.drop("Outcome", axis=1)

X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled, columns = X.columns)

###############################
# Modeling & Prediction
###############################

knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1 , random_state=45)

knn_model.predict(random_user)

###############################
# Model Evaluation
###############################
y_pred = knn_model.predict(X)

# AUC için y_prob
y_prob = knn_model.predict_proba(X)[:,1]

print(classification_report(y, y_pred))

# AUC
roc_auc_score(y, y_prob)
#0.90

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=['accuracy','f1','roc_auc'])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

###############################
# Hyperparameter Optimization
###############################

knn_params = {"n_neighbors" : range(2,50)}

knn_gs_best = GridSearchCV(knn_model, knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)
knn_gs_best.best_params_

###############################
# Final Model
###############################

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)
cv_results = cross_validate(knn_final, X, y, cv=5, scoring=['accuracy','f1','roc_auc'])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

random_user = X.sample(1)

knn_final.predict(random_user)