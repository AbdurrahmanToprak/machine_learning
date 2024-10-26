import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_validate,validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv('datasets/diabetes.csv')

y = df["Outcome"]
X = df.drop("Outcome", axis=1)

###################################
# Random Forests
###################################

rf_model = RandomForestClassifier(random_state=17)

cv_results = cross_validate(rf_model, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
#0.76
cv_results['test_f1'].mean()
#0.63
cv_results['test_roc_auc'].mean()
#0.82

rf_params = {"max_depth" : [5, 8, None],
             "max_features" : [3, 5, 7, "auto"],
             "min_samples_split" : [2, 5, 8, 15, 20],
             "n_estimators" : [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True ).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
#0.76
cv_results['test_f1'].mean()
#0.64
cv_results['test_roc_auc'].mean()
#0.82

def plot_feature_importance(model, features, num = len(X), save = False):
    feature_imp = pd.DataFrame({'Value' : model.feature_importances_, 'Feature' : features.columns})
    plt.figure(figsize = (10,10))
    sns.set(font_scale=1)
    sns.barplot(x = 'Value', y = 'Feature', data = feature_imp.sort_values(by="Value", ascending=False)[0:num])

    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('feature_importance.png')

plot_feature_importance(rf_final, X)

def val_curve_params(model, X, y, param_name, param_range, scoring = 'roc_auc', cv = 10):
    train_score, test_score = validation_curve(
        model,X, y, param_name = param_name, param_range = param_range, scoring = scoring, cv = cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score, label = 'train', color='blue')
    plt.plot(param_range, mean_test_score, label = 'test', color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"{param_name}")
    plt.ylabel(f"{scoring}")
    plt.legend(loc='best')
    plt.show()

val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring = 'f1')

###################################
# GBM
###################################

gbm_model = GradientBoostingClassifier(random_state=17)
gbm_model.get_params()

cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
#0.75
cv_results['test_f1'].mean()
#0.63
cv_results['test_roc_auc'].mean()
#0.82

gbm_params = {"max_depth" : [3, 8, 10],
             "learning_rate" : [0.01, 0.1],
             "subsample" : [1, 0.5, 0.7],
             "n_estimators" : [100, 500, 1000]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True ).fit(X, y)

gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
#0.77
cv_results['test_f1'].mean()
#0.66
cv_results['test_roc_auc'].mean()
#0.83

###################################
# XGBoost
###################################

xgboost_model = XGBClassifier(random_state=17)

cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
#0.74
cv_results['test_f1'].mean()
#0.61
cv_results['test_roc_auc'].mean()
#0.79

xgboost_params = {"max_depth" : [5, 8, None],
                  "learning_rate" : [0.1, 0.01],
                  "colsample_bytree" : [None, 0.7, 1],
                  "n_estimators" : [100, 500, 1000]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True ).fit(X, y)

xgboost_best_grid.best_params_

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
#0.76
cv_results['test_f1'].mean()
#0.64
cv_results['test_roc_auc'].mean()
#0.82

############################################
#LightGBM
###########################################

lgbm_model = LGBMClassifier(random_state=17)

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
#0.74
cv_results['test_f1'].mean()
#0.62
cv_results['test_roc_auc'].mean()
#0.79
lgbm_model.get_params()

lgbm_params = {"learning_rate" : [0.1, 0.01],
               "colsample_bytree" : [0.5, 0.7, 1],
               "n_estimators" : [100, 300, 500, 1000]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True ).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state = 17 ).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
#0.76
cv_results['test_f1'].mean()
#0.63
cv_results['test_roc_auc'].mean()
#0.81

############################################
# CatBoost
###########################################

catboost_model = CatBoostClassifier(random_state = 17, verbose = False)

cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
#0.77
cv_results['test_f1'].mean()
#0.65
cv_results['test_roc_auc'].mean()
#0.83

catboost_params = {"learning_rate" : [0.1, 0.01],
                   "depth" : [3, 6],
                   "iterations" : [200, 500]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True ).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_ , random_state= 17 ).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'] )
cv_results['test_accuracy'].mean()
#0.77
cv_results['test_f1'].mean()
#0.63
cv_results['test_roc_auc'].mean()
#0.84

############################################
# Feature Importance
#########################################

def plot_feature_importance(model, features, num = len(X), save = False):
    feature_imp = pd.DataFrame({'Value' : model.feature_importances_, 'Feature' : features.columns})
    plt.figure(figsize = (10,10))
    sns.set(font_scale=1)
    sns.barplot(x = 'Value', y = 'Feature', data = feature_imp.sort_values(by="Value", ascending=False)[0:num])

    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('feature_importance.png')

plot_feature_importance(rf_final, X)
plot_feature_importance(gbm_final, X)
plot_feature_importance(xgboost_final, X)
plot_feature_importance(lgbm_final, X)
plot_feature_importance(catboost_final, X)


############################################
# Hyperparameter Optimization with RandomSearchCV
#########################################

rf_model = RandomForestClassifier(random_state=17)

rf_random_params = {"max_depth" : np.random.randint(5, 50, 10),
                    "max_features" : [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split" : np.random.randint(5, 50, 20),
                    "n_estimators" : [int(x) for x in np.linspace(start = 200, stop = 1500, num = 10)]}

rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions = rf_random_params,
                               n_iter=100,
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)
rf_random.fit(X, y)

rf_random_final = rf_model.set_params(**rf_random.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_random_final, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
#0.76
cv_results['test_f1'].mean()
#0.62
cv_results['test_roc_auc'].mean()
#0.83

############################################
# Analyzing Model Complexity with Learning Curves
############################################

def val_curve_params(model, X, y, param_name, param_range, scoring = 'roc_auc', cv = 10):
    train_score, test_score = validation_curve(
        model,X, y, param_name = param_name, param_range = param_range, scoring = scoring, cv = cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score, label = 'train', color='blue')
    plt.plot(param_range, mean_test_score, label = 'test', color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"{param_name}")
    plt.ylabel(f"{scoring}")
    plt.legend(loc='best')
    plt.show()

rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features" , [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]

rf_model = RandomForestClassifier(random_state=17)

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])