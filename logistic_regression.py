########################################
# Diabetes prediction with logistic regression
#######################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from linear_regression import X_train, y_train, y_test

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate

def outlier_thresholds(dataframe , col_name, q1=0.05, q3=0.95):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range  = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low , up = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low, up = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] > up, variable] = up
    dataframe.loc[dataframe[variable] < low, variable] = low


#################################
# Exploratory data analysis
###############################

df = pd.read_csv('datasets/diabetes.csv')

###################################
#Target in Analizi
################################

df["Outcome"].value_counts()

sns.countplot(x="Outcome", data=df)
plt.show()

100 * df["Outcome"].value_counts() / len(df)

##################################
# Feature ların analizi
#################################

df.describe().T

df["BloodPressure"].hist(bins = 20)
plt.xlabel("BloodPressure")
plt.show()

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins = 20)
    plt.xlabel(numerical_col)
    plt.show(block=True)
for col in df.columns:
    plot_numerical_col(df, col)

cols = [col for col in df.columns if "Outcome" not in col]

for col in cols:
    plot_numerical_col(df, col)

###############################
# Target vs Feature
##############################

df.groupby("Outcome").agg({"Pregnancies": "mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)

##############################################
# Data Preprocessing (veri ön işleme)
############################################

df.shape
df.isnull().sum()
df.describe().T

for col in cols:
    print(col, check_outlier(df, col))

replace_with_thresholds(df, "Insulin")

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

##################################################
#Model & Prediction
################################################

y = df["Outcome"]

X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)

y_pred[0:10]
y[0:10]

#######################################
# Model Evaluation
######################################

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt= ".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-Score: 0.65

#ROC AUC
y_prob = log_model.predict_proba(X)[:,1]
roc_auc_score(y, y_prob)
#0.8394

#####################################
# Model Validation: Holdout
######################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)
y_pred = log_model.predict(X_test)

y_prob = log_model.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))

# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-Score: 0.65

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-Score: 0.63

#ROC AUC
RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.show()

#AUC
roc_auc_score(y_test, y_prob)

#########################################
# Model Validation: 10-Fold Cross Validation
########################################
y = df["Outcome"]

X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model, X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
#accuracy: 0.77
cv_results['test_precision'].mean()
#precision: 0.71
cv_results['test_recall'].mean()
#recall: 0.57
cv_results['test_f1'].mean()
#f1: 0.63
cv_results['test_roc_auc'].mean()
#roc_auc: 0.83


####################################
# Prediction for A New Observation
#######################################

random_user = X.sample(1, random_state=45)
log_model.predict(random_user)